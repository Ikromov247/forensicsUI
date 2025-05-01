import argparse
import sys
from pathlib import Path
import datetime

from .config import ApplicationConfig
from .application import ForensicsApplication
from .utils.input_handler import InputHandler
from .logging_config import setup_logging, get_logger

logger = get_logger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Forensics Video Analysis")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--db-name", default="forensics", help="Database name")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")
    return parser.parse_args()

def main():
    """
    Main entry point for the application.
    """
    try:
        # Parse arguments
        args = parse_args()
        
        # Check video file
        video_path = Path(args.video_path)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            sys.exit(1)
            
        # Setup logging
        log_dir = Path("logs")
        setup_logging(log_dir=log_dir)
        
        # Load configuration
        config_path = Path("config/default.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        config = ApplicationConfig.from_yaml(config_path)
        
        # Generate database name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.database.name = f"forensics_{timestamp}.db"
        
        # Initialize application
        app = ForensicsApplication(
            config=config,
            db_name=args.db_name,
            visualization=not args.no_vis
        )
        
        # Get input files
        input_handler = InputHandler()
        image_path, _ = input_handler.get_input_files()
        
        # Process target image
        target_data = app.process_target_image(str(image_path))
        
        # Process video
        results = app.process_video(str(video_path), target_data)
        
        # Save results to database
        app.db_manager.save_results(results)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        sys.exit(1)
        
    finally:
        if 'app' in locals():
            app.cleanup()

if __name__ == "__main__":
    sys.exit(main()) 