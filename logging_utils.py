import os
import json
import logging
from datetime import datetime
from pathlib import Path
import torch


class ExperimentLogger:
    def __init__(self, log_dir="./logs", experiment_name=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"experiment_{timestamp}"
        else:
            self.experiment_name = experiment_name
            
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        self.log_file = self.experiment_dir / "training.log"
        self.setup_logging()
        
        self.results = {
            "hyperparameters": {},
            "training_history": [],
            "best_results": {},
            "final_results": {}
        }
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_hyperparameters(self, args):
        hyperparams = vars(args)
        self.results["hyperparameters"] = hyperparams
        
        self.logger.info("=" * 50)
        self.logger.info("EXPERIMENT HYPERPARAMETERS")
        self.logger.info("=" * 50)
        for key, value in hyperparams.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 50)
        
        with open(self.experiment_dir / "hyperparameters.json", "w") as f:
            json.dump(hyperparams, f, indent=2)
            
    def log_epoch_results(self, epoch, train_loss, dev_accuracy, test_clean_accuracy=None, test_poison_accuracy=None, asr=None):
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_accuracy": dev_accuracy,
            "test_clean_accuracy": test_clean_accuracy,
            "test_poison_accuracy": test_poison_accuracy,
            "asr": asr
        }
        
        self.results["training_history"].append(epoch_result)
        
        self.logger.info(f"Epoch {epoch}:")
        self.logger.info(f"  Train Loss: {train_loss:.4f}")
        self.logger.info(f"  Dev Accuracy: {dev_accuracy:.4f}")
        if test_clean_accuracy is not None:
            self.logger.info(f"  Test Clean Accuracy: {test_clean_accuracy:.4f}")
        if test_poison_accuracy is not None:
            self.logger.info(f"  Test Poison Accuracy: {test_poison_accuracy:.4f}")
        if asr is not None:
            self.logger.info(f"  Attack Success Rate (ASR): {asr:.4f}")
        self.logger.info("-" * 30)
        
    def log_best_results(self, best_dev_acc, best_test_clean_acc, best_test_poison_acc, best_asr):
        """记录最佳结果"""
        self.results["best_results"] = {
            "best_dev_accuracy": best_dev_acc,
            "best_test_clean_accuracy": best_test_clean_acc,
            "best_test_poison_accuracy": best_test_poison_acc,
            "best_asr": best_asr
        }
        
        self.logger.info("=" * 50)
        self.logger.info("BEST RESULTS")
        self.logger.info("=" * 50)
        self.logger.info(f"Best Dev Accuracy: {best_dev_acc:.4f}")
        self.logger.info(f"Best Test Clean Accuracy: {best_test_clean_acc:.4f}")
        self.logger.info(f"Best Test Poison Accuracy: {best_test_poison_acc:.4f}")
        self.logger.info(f"Best Attack Success Rate (ASR): {best_asr:.4f}")
        self.logger.info("=" * 50)
        
    def log_system_info(self):

        self.logger.info("=" * 50)
        self.logger.info("SYSTEM INFORMATION")
        self.logger.info("=" * 50)
        self.logger.info(f"PyTorch version: {torch.__version__}")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            self.logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            self.logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
        self.logger.info("=" * 50)
        
    def log_experiment_start(self):

        self.logger.info("=" * 50)
        self.logger.info(f"EXPERIMENT STARTED: {self.experiment_name}")
        self.logger.info(f"Log directory: {self.experiment_dir}")
        self.logger.info("=" * 50)
        
    def log_experiment_end(self):

        self.logger.info("=" * 50)
        self.logger.info(f"EXPERIMENT COMPLETED: {self.experiment_name}")
        self.logger.info("=" * 50)
        

        with open(self.experiment_dir / "results.json", "w") as f:
            json.dump(self.results, f, indent=2)
            
    def log_model_save(self, model_path):

        self.logger.info(f"Model saved to: {model_path}")
        
    def log_error(self, error_msg):

        self.logger.error(f"ERROR: {error_msg}")
        
    def log_warning(self, warning_msg):

        self.logger.warning(f"WARNING: {warning_msg}")


def create_logger(args, experiment_name=None):

    if experiment_name is None:
        model_name = args.model_name_or_path.split('/')[-1]
        poison_str = f"_poison_{args.poison}" if args.poison else ""
        experiment_name = f"{model_name}_method{args.method}{poison_str}_lr{args.learning_rate}_bs{args.per_device_train_batch_size}"
    
    logger = ExperimentLogger(experiment_name=experiment_name)
    return logger 
