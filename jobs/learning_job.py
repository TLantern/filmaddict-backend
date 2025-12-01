import logging
import os
from datetime import datetime, timedelta
from typing import Dict

from sqlalchemy.ext.asyncio import AsyncSession

from db import crud
from utils.learning import (
    update_calibration_from_feedback,
    evaluate_prompt_performance,
    generate_improved_prompt,
    compare_prompt_versions,
    extract_feedback_patterns,
    MIN_FEEDBACK_THRESHOLD,
    MIN_FEEDBACK_FOR_NEW_PROMPT,
)
from utils.learning_logger import (
    log_calibration_change,
    log_prompt_version_change,
    log_feedback_pattern_change,
    log_future_change,
    mark_future_change_completed,
)

logger = logging.getLogger(__name__)

LEARNING_DAYS = int(os.getenv("LEARNING_DAYS", "7"))


async def run_learning_pipeline(db: AsyncSession) -> Dict[str, any]:
    """
    Main learning pipeline that:
    1. Collects feedback from last N days
    2. Calculates calibration offset if sufficient data
    3. Evaluates all prompt versions' performance based on editor feedback
    4. Generates new prompt variant if enough data
    5. Promotes best performing prompt version if statistically significant improvement
    
    Prompt promotion is based purely on editor feedback (ratings, saves, positive/negative flags),
    not viewer CTR or other view-based metrics.
    
    Args:
        db: Database session
        
    Returns:
        Dictionary with results of learning pipeline execution
    """
    result = {
        "calibration_updated": False,
        "prompt_evaluated": False,
        "prompt_promoted": False,
        "new_prompt_created": False,
    }
    
    try:
        logger.info("Starting learning pipeline")
        
        # Step 1: Update calibration from feedback
        try:
            # Get current calibration config for logging
            current_config = await crud.get_calibration_config(db)
            previous_offset = current_config.score_offset if current_config else None
            
            offset = await update_calibration_from_feedback(db, days=LEARNING_DAYS)
            if offset is not None:
                result["calibration_updated"] = True
                logger.info(f"Calibration updated with offset: {offset:.2f}")
                
                # Log calibration change
                updated_config = await crud.get_calibration_config(db)
                if updated_config:
                    log_calibration_change(
                        previous_offset=previous_offset,
                        new_offset=offset,
                        sample_size=updated_config.sample_size,
                        feedback_count=updated_config.feedback_count,
                        change_type="update" if previous_offset is not None else "initial",
                    )
        except Exception as e:
            logger.error(f"Error updating calibration: {str(e)}", exc_info=True)
        
        # Step 2: Evaluate all prompt versions
        try:
            prompt_versions = await crud.get_all_prompt_versions(db)
            
            if not prompt_versions:
                logger.info("No prompt versions found, skipping evaluation")
                return result
            
            result["prompt_evaluated"] = True
            best_version = None
            best_metrics = None
            
            for version in prompt_versions:
                metrics = await evaluate_prompt_performance(db, version.id, days=LEARNING_DAYS)
                previous_metrics = {
                    "avg_rating": version.avg_rating,
                    "save_rate": version.save_rate,
                    "positive_rate": version.positive_rate,
                    "negative_rate": version.negative_rate,
                    "sample_size": version.total_rated,
                }
                await crud.update_prompt_version_metrics(db, version.id, metrics)
                
                logger.info(
                    f"Prompt version {version.version_name}: "
                    f"avg_rating={metrics['avg_rating']:.2f}, "
                    f"save_rate={metrics.get('save_rate', 0.0):.2f}, "
                    f"positive_rate={metrics.get('positive_rate', 0.0):.2f}, "
                    f"sample_size={metrics['sample_size']}"
                )
                
                # Log prompt evaluation
                log_prompt_version_change(
                    version_id=version.id,
                    version_name=version.version_name,
                    change_type="evaluated",
                    previous_state=previous_metrics,
                    new_state=metrics,
                    metrics=metrics,
                )
                
                if metrics["sample_size"] >= 50:
                    if best_version is None or metrics["avg_rating"] > best_metrics["avg_rating"]:
                        best_version = version
                        best_metrics = metrics
            
            # Step 3: Generate new prompt variant if enough data
            try:
                feedback_list = await crud.get_recent_feedback(db, days=LEARNING_DAYS)
                logger.info(f"Found {len(feedback_list)} feedback items for prompt generation (minimum: {MIN_FEEDBACK_FOR_NEW_PROMPT})")
                
                # Log feedback pattern analysis
                if feedback_list:
                    patterns = extract_feedback_patterns(feedback_list)
                    log_feedback_pattern_change(
                        patterns=patterns,
                        feedback_count=len(feedback_list),
                        change_type="analysis",
                    )
                
                if len(feedback_list) >= MIN_FEEDBACK_FOR_NEW_PROMPT:
                    active_version = await crud.get_active_prompt_version(db)
                    
                    if active_version:
                        logger.info(f"Generating improved prompt from {len(feedback_list)} feedback items...")
                        new_prompt_template = generate_improved_prompt(
                            feedback_list,
                            active_version.user_prompt_template,
                        )
                        
                        if new_prompt_template != active_version.user_prompt_template:
                            version_name = f"v{len(prompt_versions) + 1}"
                            new_version = await crud.create_prompt_version(
                                db,
                                version_name=version_name,
                                system_prompt=active_version.system_prompt,
                                user_prompt_template=new_prompt_template,
                                is_active=False,
                            )
                            result["new_prompt_created"] = True
                            logger.info(f"Created new prompt version: {version_name}")
                            
                            # Log prompt creation
                            log_prompt_version_change(
                                version_id=new_version.id,
                                version_name=version_name,
                                change_type="created",
                                new_state={
                                    "system_prompt": new_version.system_prompt[:100] + "..." if len(new_version.system_prompt) > 100 else new_version.system_prompt,
                                    "user_prompt_template": new_version.user_prompt_template[:200] + "..." if len(new_version.user_prompt_template) > 200 else new_version.user_prompt_template,
                                    "is_active": new_version.is_active,
                                },
                            )
                        else:
                            logger.info("Prompt generation did not produce changes (feedback patterns insufficient)")
                    else:
                        logger.info("No active prompt version found, skipping prompt generation")
                else:
                    logger.info(f"Insufficient feedback for prompt generation: {len(feedback_list)} < {MIN_FEEDBACK_FOR_NEW_PROMPT}")
            except Exception as e:
                logger.error(f"Error generating new prompt: {str(e)}", exc_info=True)
            
            # Step 4: Promote best performing version if significant improvement
            if best_version and best_metrics:
                active_version = await crud.get_active_prompt_version(db)
                
                if active_version and active_version.id != best_version.id:
                    active_metrics = await evaluate_prompt_performance(db, active_version.id, days=LEARNING_DAYS)
                    
                    comparison = compare_prompt_versions(active_metrics, best_metrics)
                    
                    if comparison["significant"] and comparison["winner"] == "B":
                        previous_state = {
                            "version_id": str(active_version.id),
                            "version_name": active_version.version_name,
                            "is_active": active_version.is_active,
                            "metrics": active_metrics,
                        }
                        await crud.activate_prompt_version(db, best_version.id)
                        result["prompt_promoted"] = True
                        logger.info(
                            f"Promoted prompt version {best_version.version_name} "
                            f"(improvement: {comparison['improvement']['rating_improvement']:.2f})"
                        )
                        
                        # Log prompt promotion
                        log_prompt_version_change(
                            version_id=best_version.id,
                            version_name=best_version.version_name,
                            change_type="promoted",
                            previous_state=previous_state,
                            new_state={
                                "version_id": str(best_version.id),
                                "version_name": best_version.version_name,
                                "is_active": True,
                                "metrics": best_metrics,
                            },
                            metrics=best_metrics,
                            comparison=comparison,
                        )
                        
                        # Mark any future changes for this promotion as completed
                        mark_future_change_completed("prompt_promotion")
                elif not active_version:
                    await crud.activate_prompt_version(db, best_version.id)
                    result["prompt_promoted"] = True
                    logger.info(f"Activated first prompt version: {best_version.version_name}")
                    
                    # Log prompt activation
                    log_prompt_version_change(
                        version_id=best_version.id,
                        version_name=best_version.version_name,
                        change_type="activated",
                        new_state={
                            "version_id": str(best_version.id),
                            "version_name": best_version.version_name,
                            "is_active": True,
                            "metrics": best_metrics,
                        },
                        metrics=best_metrics,
                    )
        
        except Exception as e:
            logger.error(f"Error evaluating prompts: {str(e)}", exc_info=True)
        
        logger.info("Learning pipeline completed")
        return result
    
    except Exception as e:
        logger.error(f"Error in learning pipeline: {str(e)}", exc_info=True)
        raise

