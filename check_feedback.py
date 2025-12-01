#!/usr/bin/env python3
"""Script to check current feedback in the database."""
import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv

from database import async_session_maker
from db.models import HighlightFeedback, Highlight
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from models import FeedbackType

load_dotenv()


async def check_feedback():
    """Query and display all feedback records."""
    async with async_session_maker() as db:
        # Get all feedback with highlights eagerly loaded
        result = await db.execute(
            select(HighlightFeedback)
            .options(selectinload(HighlightFeedback.highlight))
            .order_by(HighlightFeedback.created_at.desc())
        )
        all_feedback = list(result.scalars().all())
        
        # Get statistics
        stats_result = await db.execute(
            select(
                func.count(HighlightFeedback.id).label('total'),
                func.count(HighlightFeedback.rating).label('ratings'),
                func.avg(HighlightFeedback.rating).label('avg_rating'),
                func.count(HighlightFeedback.text_feedback).label('text_feedback_count')
            )
        )
        stats = stats_result.first()
        
        print("=" * 80)
        print("FEEDBACK SUMMARY")
        print("=" * 80)
        print(f"Total feedback records: {stats.total}")
        print(f"Records with ratings: {stats.ratings}")
        print(f"Average rating: {stats.avg_rating:.2f}" if stats.avg_rating else "Average rating: N/A")
        print(f"Records with text feedback: {stats.text_feedback_count}")
        print()
        
        if not all_feedback:
            print("No feedback records found in the database.")
            return
        
        print("=" * 80)
        print("RECENT FEEDBACK (Last 20)")
        print("=" * 80)
        
        for i, feedback in enumerate(all_feedback[:20], 1):
            print(f"\n{i}. Feedback ID: {feedback.id}")
            print(f"   Type: {feedback.feedback_type}")
            print(f"   Rating: {feedback.rating}" if feedback.rating else "   Rating: None")
            print(f"   Text: {feedback.text_feedback[:100] + '...' if feedback.text_feedback and len(feedback.text_feedback) > 100 else feedback.text_feedback or 'None'}")
            print(f"   Created: {feedback.created_at}")
            if feedback.highlight:
                print(f"   Highlight: {feedback.highlight.start:.1f}s - {feedback.highlight.end:.1f}s")
                print(f"   Reason: {feedback.highlight.reason[:80] + '...' if len(feedback.highlight.reason) > 80 else feedback.highlight.reason}")
        
        if len(all_feedback) > 20:
            print(f"\n... and {len(all_feedback) - 20} more feedback records")
        
        # Group by feedback type
        print("\n" + "=" * 80)
        print("FEEDBACK BY TYPE")
        print("=" * 80)
        type_counts = {}
        for feedback in all_feedback:
            type_counts[feedback.feedback_type] = type_counts.get(feedback.feedback_type, 0) + 1
        
        for feedback_type, count in sorted(type_counts.items()):
            print(f"{feedback_type}: {count}")
        
        # Show ratings distribution
        ratings = [f.rating for f in all_feedback if f.rating is not None]
        if ratings:
            print("\n" + "=" * 80)
            print("RATING DISTRIBUTION")
            print("=" * 80)
            print(f"Min: {min(ratings):.1f}")
            print(f"Max: {max(ratings):.1f}")
            print(f"Average: {sum(ratings) / len(ratings):.1f}")
            
            # Count by ranges
            high = len([r for r in ratings if r >= 70])
            medium = len([r for r in ratings if 40 <= r < 70])
            low = len([r for r in ratings if r < 40])
            print(f"\nHigh (≥70): {high}")
            print(f"Medium (40-69): {medium}")
            print(f"Low (<40): {low}")


if __name__ == "__main__":
    asyncio.run(check_feedback())

