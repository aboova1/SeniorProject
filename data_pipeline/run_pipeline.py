#!/usr/bin/env python3
"""
Senior Project Data Pipeline Runner
Executes all data collection and processing steps in the correct order
"""

import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(self):
        self.pipeline_dir = Path(__file__).parent
        self.steps = [
            {
                'name': 'Data Collection',
                'script': 'collect_sp500_data.py',
                'description': 'Downloading S&P 500 membership, prices, fundamentals, and earnings data'
            },
            {
                'name': 'Earnings Dates Collection',
                'script': 'collect_earnings_dates.py',
                'description': 'Collecting actual earnings announcement dates from FMP (optional, enhances accuracy)',
                'optional': True
            },
            {
                'name': 'Transcript Timing Analysis',
                'script': 'calculate_transcript_timing.py',
                'description': 'Calculating company-specific transcript release delays for optimal rebalance dates'
            },
            {
                'name': 'Feature Engineering',
                'script': 'engineer_features.py',
                'description': 'Computing technical indicators, fundamental ratios, and PCA-reduced embeddings'
            },
            {
                'name': 'Transcript Filtering',
                'script': 'filter_by_transcripts.py',
                'description': 'Filtering to quarters with transcript coverage'
            },
            {
                'name': 'Sequence Creation',
                'script': 'create_sequences.py',
                'description': 'Creating overlapping 8-quarter sequences for model training'
            }
        ]
        self.start_time = None
        self.results = []

    def run_script(self, script_path: Path, step_name: str) -> bool:
        """Run a single Python script and return success status"""
        try:
            logger.info("="*80)
            logger.info(f"STEP: {step_name}")
            logger.info(f"Script: {script_path.name}")
            logger.info("="*80)

            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(script_path.parent),
                capture_output=False,
                text=True,
                check=True
            )

            logger.info(f"✓ {step_name} completed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {step_name} failed with exit code {e.returncode}")
            return False
        except Exception as e:
            logger.error(f"✗ {step_name} failed with error: {e}")
            return False

    def run_pipeline(self, start_from: int = 0, skip_steps: list = None):
        """
        Run the full pipeline or start from a specific step

        Args:
            start_from: Step number to start from (0-indexed)
            skip_steps: List of step indices to skip
        """
        if skip_steps is None:
            skip_steps = []

        self.start_time = datetime.now()
        logger.info("="*80)
        logger.info("STARTING DATA PIPELINE")
        logger.info(f"Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total steps: {len(self.steps)}")
        if start_from > 0:
            logger.info(f"Starting from step {start_from + 1}: {self.steps[start_from]['name']}")
        if skip_steps:
            logger.info(f"Skipping steps: {[i+1 for i in skip_steps]}")
        logger.info("="*80)
        logger.info("")

        for i, step in enumerate(self.steps):
            if i < start_from:
                logger.info(f"⊘ Skipping step {i+1}/{len(self.steps)}: {step['name']}")
                self.results.append({'step': step['name'], 'status': 'SKIPPED'})
                continue

            if i in skip_steps:
                logger.info(f"⊘ Skipping step {i+1}/{len(self.steps)}: {step['name']} (user requested)")
                self.results.append({'step': step['name'], 'status': 'SKIPPED'})
                continue

            logger.info(f"\n▶ Step {i+1}/{len(self.steps)}: {step['name']}")
            logger.info(f"  {step['description']}")

            script_path = self.pipeline_dir / step['script']
            if not script_path.exists():
                logger.error(f"  Script not found: {script_path}")
                self.results.append({'step': step['name'], 'status': 'FAILED - Script not found'})

                # If it's an optional step, continue
                if step.get('optional', False):
                    logger.warning(f"  Step is optional, continuing pipeline...")
                    continue
                else:
                    break

            step_start = datetime.now()
            success = self.run_script(script_path, step['name'])
            step_duration = (datetime.now() - step_start).total_seconds()

            if success:
                logger.info(f"  Duration: {step_duration:.1f}s")
                self.results.append({'step': step['name'], 'status': 'SUCCESS', 'duration': step_duration})
            else:
                logger.error(f"  Failed after {step_duration:.1f}s")
                self.results.append({'step': step['name'], 'status': 'FAILED', 'duration': step_duration})

                # If it's an optional step, continue
                if step.get('optional', False):
                    logger.warning(f"  Step is optional, continuing pipeline...")
                    continue
                else:
                    logger.error("\n❌ Pipeline stopped due to error")
                    break

        self.print_summary()

    def print_summary(self):
        """Print pipeline execution summary"""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()

        logger.info("\n" + "="*80)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*80)

        for i, result in enumerate(self.results):
            status_symbol = "✓" if result['status'] == 'SUCCESS' else "✗" if result['status'] == 'FAILED' else "⊘"
            duration_str = f"({result.get('duration', 0):.1f}s)" if 'duration' in result else ""
            logger.info(f"{status_symbol} Step {i+1}: {result['step']} - {result['status']} {duration_str}")

        logger.info("-"*80)
        success_count = sum(1 for r in self.results if r['status'] == 'SUCCESS')
        failed_count = sum(1 for r in self.results if 'FAILED' in r['status'])
        skipped_count = sum(1 for r in self.results if r['status'] == 'SKIPPED')

        logger.info(f"Total steps: {len(self.results)}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Skipped: {skipped_count}")
        logger.info(f"Total duration: {total_duration/60:.1f} minutes")
        logger.info("="*80)

        if failed_count == 0 and success_count > 0:
            logger.info("\n🎉 Pipeline completed successfully!")
            logger.info("Output file: data_pipeline/data/sequences_8q.parquet")
        elif failed_count > 0:
            logger.error("\n❌ Pipeline completed with errors")
            logger.error("Check the logs above for details")


def main():
    """Main entry point with command-line options"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run the Senior Project data pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_pipeline.py

  # Start from feature engineering (skip data collection)
  python run_pipeline.py --start-from 3

  # Skip transcript collection (if already done)
  python run_pipeline.py --skip 1

  # Skip embedding generation (if already done)
  python run_pipeline.py --skip 2

  # Start from filtering and skip sequence creation
  python run_pipeline.py --start-from 3 --skip 4

Step numbers:
  1: collect_sp500_data.py
  2: engineer_features.py
  3: filter_by_transcripts.py
  4: create_sequences.py
        """
    )

    parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        metavar='N',
        help='Start from step N (0-5)'
    )

    parser.add_argument(
        '--skip',
        type=int,
        nargs='+',
        default=[],
        metavar='N',
        help='Skip step(s) N (0-5), can specify multiple'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.start_from < 0 or args.start_from > 5:
        parser.error("--start-from must be between 0 and 5")

    if any(s < 0 or s > 5 for s in args.skip):
        parser.error("--skip values must be between 0 and 5")

    # Run pipeline
    runner = PipelineRunner()
    runner.run_pipeline(start_from=args.start_from, skip_steps=args.skip)


if __name__ == "__main__":
    main()
