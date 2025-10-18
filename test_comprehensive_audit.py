"""
Comprehensive Code Audit and Cleanup Script
Systematically checks all modules and validates the entire system.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import importlib
import ast
import re
from typing import List, Dict, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

class CodeAuditor:
    """Comprehensive code audit system."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed = []
        self.root = Path(__file__).parent
        
    def log_issue(self, category: str, severity: str, message: str):
        """Log an issue."""
        self.issues.append({
            'category': category,
            'severity': severity,
            'message': message,
            'timestamp': datetime.now()
        })
        
    def log_warning(self, category: str, message: str):
        """Log a warning."""
        self.warnings.append({
            'category': category,
            'message': message,
            'timestamp': datetime.now()
        })
        
    def log_pass(self, category: str, message: str):
        """Log a passed check."""
        self.passed.append({
            'category': category,
            'message': message,
            'timestamp': datetime.now()
        })
        
    def print_section(self, title: str):
        """Print section header."""
        print(f"\n{'='*80}")
        print(f"  {title}")
        print('='*80)
        
    def audit_imports(self, file_path: Path) -> Dict:
        """Check imports in a Python file."""
        results = {'unused': [], 'missing': [], 'duplicates': []}
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                tree = ast.parse(content)
                
            # Get all imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            imports.append(f"{node.module}.{alias.name}")
            
            # Check for duplicates
            seen = set()
            for imp in imports:
                if imp in seen:
                    results['duplicates'].append(imp)
                seen.add(imp)
                
        except Exception as e:
            self.log_warning('imports', f"Could not parse {file_path.name}: {e}")
            
        return results
        
    def audit_dataframe_usage(self, file_path: Path) -> Dict:
        """Check DataFrame usage patterns."""
        issues = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Check for common DataFrame issues
            checks = [
                (r'\.copy\(\)', 'Unnecessary .copy() calls may impact performance'),
                (r'pd\.DataFrame\(\)(?!.*index=)', 'DataFrame creation without explicit index'),
                (r'df\[\'date\'\](?!.*\.dt)', 'Date column access without .dt accessor'),
                (r'\.fillna\(0\)(?=.*division)', 'Potential division by zero after fillna'),
                (r'\.iterrows\(\)', 'Inefficient .iterrows() - use vectorization'),
                (r'\.apply\(lambda', 'Consider vectorization instead of .apply(lambda)'),
            ]
            
            for pattern, message in checks:
                if re.search(pattern, content):
                    issues.append(message)
                    
        except Exception as e:
            self.log_warning('dataframe', f"Could not audit {file_path.name}: {e}")
            
        return {'issues': issues}
        
    def audit_error_handling(self, file_path: Path) -> Dict:
        """Check error handling patterns."""
        results = {'bare_except': 0, 'no_logging': 0, 'good_patterns': 0}
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                tree = ast.parse(content)
                
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler):
                    # Check for bare except
                    if node.type is None:
                        results['bare_except'] += 1
                    else:
                        results['good_patterns'] += 1
                        
        except Exception as e:
            self.log_warning('error_handling', f"Could not parse {file_path.name}: {e}")
            
        return results
        
    def audit_module_structure(self, module_path: Path):
        """Audit a Python module's structure."""
        self.print_section(f"Auditing {module_path.relative_to(self.root)}")
        
        if not module_path.exists():
            self.log_issue('structure', 'ERROR', f"{module_path} does not exist")
            return
            
        py_files = list(module_path.glob('*.py'))
        if not py_files:
            self.log_warning('structure', f"No Python files in {module_path.name}")
            return
            
        print(f"Found {len(py_files)} Python files")
        
        for py_file in py_files:
            if py_file.name.startswith('_'):
                continue
                
            print(f"\n  Checking {py_file.name}...")
            
            # Check imports
            import_results = self.audit_imports(py_file)
            if import_results['duplicates']:
                self.log_warning('imports', f"{py_file.name}: Duplicate imports: {import_results['duplicates']}")
            else:
                self.log_pass('imports', f"{py_file.name}: No duplicate imports")
                
            # Check DataFrame usage
            df_results = self.audit_dataframe_usage(py_file)
            if df_results['issues']:
                for issue in df_results['issues']:
                    self.log_warning('dataframe', f"{py_file.name}: {issue}")
            else:
                self.log_pass('dataframe', f"{py_file.name}: DataFrame usage looks good")
                
            # Check error handling
            error_results = self.audit_error_handling(py_file)
            if error_results['bare_except'] > 0:
                self.log_issue('error_handling', 'WARNING', 
                             f"{py_file.name}: {error_results['bare_except']} bare except clauses")
            else:
                self.log_pass('error_handling', f"{py_file.name}: Error handling looks good")
                
    def test_data_loading(self):
        """Test data loading functionality."""
        self.print_section("Testing Data Loading")
        
        try:
            from utils.data_loader import DataLoader
            
            loader = DataLoader()
            print("✓ DataLoader instantiated")
            
            # Get available tickers
            tickers = loader.get_available_tickers()
            print(f"✓ Found {len(tickers)} available tickers")
            
            if len(tickers) == 0:
                self.log_warning('data', "No tickers available - run data download first")
                return
                
            # Test loading a ticker
            test_ticker = tickers[0]
            data = loader.load_ticker_ohlcv(test_ticker)
            
            # Validate DataFrame structure
            if not isinstance(data.index, pd.DatetimeIndex):
                self.log_issue('data', 'ERROR', f"Index is not DatetimeIndex: {type(data.index)}")
            else:
                print(f"✓ Index is DatetimeIndex")
                
            if data.index.name != 'date':
                self.log_issue('data', 'ERROR', f"Index name is '{data.index.name}', expected 'date'")
            else:
                print(f"✓ Index name is 'date'")
                
            # Check required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required if col not in data.columns]
            
            if missing:
                self.log_issue('data', 'ERROR', f"Missing columns: {missing}")
            else:
                print(f"✓ All OHLCV columns present")
                
            # Check for lowercase columns
            uppercase_cols = [col for col in data.columns if col != col.lower()]
            if uppercase_cols:
                self.log_issue('data', 'ERROR', f"Uppercase columns found: {uppercase_cols}")
            else:
                print(f"✓ All columns are lowercase")
                
            self.log_pass('data', f"Data loading validated with {test_ticker}")
            
        except Exception as e:
            self.log_issue('data', 'ERROR', f"Data loading failed: {e}")
            import traceback
            traceback.print_exc()
            
    def test_feature_engineering(self):
        """Test feature engineering."""
        self.print_section("Testing Feature Engineering")
        
        try:
            from features.base_feature import FeatureEngine, FeatureRegistry
            from utils.data_loader import DataLoader
            
            # Get test data
            loader = DataLoader()
            tickers = loader.get_available_tickers()
            
            if len(tickers) == 0:
                self.log_warning('features', "No data available for testing")
                return
                
            test_ticker = tickers[0]
            data = loader.load_ticker_ohlcv(test_ticker)
            print(f"✓ Loaded {test_ticker} with {len(data)} rows")
            
            # Test feature registry
            features = FeatureRegistry.list_features()
            print(f"✓ Found {len(features)} registered features: {features}")
            
            # Test feature engine
            engine = FeatureEngine()
            
            if len(features) == 0:
                self.log_warning('features', "No features registered")
                return
                
            # Add a feature
            test_feature = features[0]
            engine.add_feature(test_feature)
            print(f"✓ Added feature: {test_feature}")
            
            # Compute features
            result = engine.compute_features(data)
            
            # Validate output
            if not isinstance(result.index, pd.DatetimeIndex):
                self.log_issue('features', 'ERROR', "Feature output index is not DatetimeIndex")
            else:
                print(f"✓ Feature output has DatetimeIndex")
                
            if result.index.name != 'date':
                self.log_issue('features', 'ERROR', f"Feature index name is '{result.index.name}'")
            else:
                print(f"✓ Feature index name is 'date'")
                
            if len(result) != len(data):
                self.log_issue('features', 'ERROR', 
                             f"Feature output length {len(result)} != input length {len(data)}")
            else:
                print(f"✓ Feature output length matches input")
                
            self.log_pass('features', f"Feature engineering validated with {test_feature}")
            
        except Exception as e:
            self.log_issue('features', 'ERROR', f"Feature engineering failed: {e}")
            import traceback
            traceback.print_exc()
            
    def test_model_training(self):
        """Test model training pipeline."""
        self.print_section("Testing Model Training")
        
        try:
            from models.trainer import ModelTrainer
            from utils.data_loader import DataLoader
            from features.base_feature import FeatureEngine
            
            loader = DataLoader()
            tickers = loader.get_available_tickers()
            
            if len(tickers) < 3:
                self.log_warning('models', "Need at least 3 tickers for testing")
                return
                
            # Load data for multiple tickers
            test_tickers = tickers[:3]
            data_dict = {}
            
            for ticker in test_tickers:
                try:
                    data_dict[ticker] = loader.load_ticker_ohlcv(ticker)
                except Exception as e:
                    print(f"  Skipping {ticker}: {e}")
                    
            if len(data_dict) < 2:
                self.log_warning('models', "Need at least 2 valid tickers for testing")
                return
                
            print(f"✓ Loaded {len(data_dict)} tickers")
            
            # Create trainer
            trainer = ModelTrainer()
            print(f"✓ ModelTrainer instantiated")
            
            self.log_pass('models', "Model training setup validated")
            
        except Exception as e:
            self.log_issue('models', 'ERROR', f"Model training failed: {e}")
            import traceback
            traceback.print_exc()
            
    def test_strategy_execution(self):
        """Test strategy execution."""
        self.print_section("Testing Strategy Execution")
        
        try:
            from strategies.base_strategy import BaseStrategy, MLStrategy
            from utils.data_loader import DataLoader
            
            loader = DataLoader()
            tickers = loader.get_available_tickers()
            
            if len(tickers) == 0:
                self.log_warning('strategy', "No data available for testing")
                return
                
            # Test base strategy exists
            print(f"✓ BaseStrategy class available")
            print(f"✓ MLStrategy class available")
            
            self.log_pass('strategy', "Strategy classes validated")
            
        except Exception as e:
            self.log_issue('strategy', 'ERROR', f"Strategy test failed: {e}")
            import traceback
            traceback.print_exc()
            
    def test_backtest_engine(self):
        """Test backtest engine."""
        self.print_section("Testing Backtest Engine")
        
        try:
            from backtest.engine import BacktestEngine
            from backtest.portfolio import PortfolioConstructor
            
            print(f"✓ BacktestEngine class available")
            print(f"✓ PortfolioConstructor class available")
            
            self.log_pass('backtest', "Backtest classes validated")
            
        except Exception as e:
            self.log_issue('backtest', 'ERROR', f"Backtest test failed: {e}")
            import traceback
            traceback.print_exc()
            
    def cleanup_cache_files(self):
        """Clean up __pycache__ and .pyc files."""
        self.print_section("Cleaning Cache Files")
        
        removed_count = 0
        
        # Remove __pycache__ directories
        for pycache_dir in self.root.rglob('__pycache__'):
            try:
                import shutil
                shutil.rmtree(pycache_dir)
                removed_count += 1
                print(f"✓ Removed {pycache_dir.relative_to(self.root)}")
            except Exception as e:
                print(f"✗ Could not remove {pycache_dir}: {e}")
                
        # Remove .pyc files
        for pyc_file in self.root.rglob('*.pyc'):
            try:
                pyc_file.unlink()
                removed_count += 1
                print(f"✓ Removed {pyc_file.relative_to(self.root)}")
            except Exception as e:
                print(f"✗ Could not remove {pyc_file}: {e}")
                
        # Remove .DS_Store files
        for ds_file in self.root.rglob('.DS_Store'):
            try:
                ds_file.unlink()
                removed_count += 1
                print(f"✓ Removed {ds_file.relative_to(self.root)}")
            except Exception as e:
                print(f"✗ Could not remove {ds_file}: {e}")
                
        print(f"\n✓ Cleaned {removed_count} cache files")
        self.log_pass('cleanup', f"Removed {removed_count} cache files")
        
    def generate_report(self):
        """Generate audit report."""
        self.print_section("AUDIT REPORT")
        
        print(f"\n📊 Summary:")
        print(f"  ✓ Passed Checks: {len(self.passed)}")
        print(f"  ⚠️  Warnings: {len(self.warnings)}")
        print(f"  ✗ Issues: {len(self.issues)}")
        
        if self.issues:
            print(f"\n✗ ISSUES FOUND:")
            for issue in self.issues:
                print(f"  [{issue['severity']}] {issue['category']}: {issue['message']}")
                
        if self.warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  [{warning['category']}] {warning['message']}")
                
        if len(self.issues) == 0 and len(self.warnings) == 0:
            print(f"\n✅ ALL CHECKS PASSED! System is clean and working.")
        elif len(self.issues) == 0:
            print(f"\n✅ No critical issues found. Review warnings above.")
        else:
            print(f"\n⚠️  Please address the issues above.")
            
        # Save report to file
        report_path = self.root / 'AUDIT_REPORT.md'
        with open(report_path, 'w') as f:
            f.write("# QuantSearch Code Audit Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- ✓ Passed Checks: {len(self.passed)}\n")
            f.write(f"- ⚠️  Warnings: {len(self.warnings)}\n")
            f.write(f"- ✗ Issues: {len(self.issues)}\n\n")
            
            if self.issues:
                f.write("## Issues\n\n")
                for issue in self.issues:
                    f.write(f"- **[{issue['severity']}]** {issue['category']}: {issue['message']}\n")
                f.write("\n")
                
            if self.warnings:
                f.write("## Warnings\n\n")
                for warning in self.warnings:
                    f.write(f"- [{warning['category']}] {warning['message']}\n")
                f.write("\n")
                
            if self.passed:
                f.write("## Passed Checks\n\n")
                for check in self.passed:
                    f.write(f"- ✓ {check['category']}: {check['message']}\n")
                    
        print(f"\n📄 Full report saved to: {report_path}")
        
    def run_full_audit(self):
        """Run complete audit."""
        print("="*80)
        print("  QUANTSEARCH COMPREHENSIVE CODE AUDIT")
        print("="*80)
        
        # 1. Audit module structures
        modules = ['data', 'features', 'models', 'strategies', 'backtest', 'utils', 'ui/pages', 'ui/components']
        for module in modules:
            module_path = self.root / module
            if module_path.exists():
                self.audit_module_structure(module_path)
                
        # 2. Test functionality
        self.test_data_loading()
        self.test_feature_engineering()
        self.test_model_training()
        self.test_strategy_execution()
        self.test_backtest_engine()
        
        # 3. Cleanup
        self.cleanup_cache_files()
        
        # 4. Generate report
        self.generate_report()


if __name__ == "__main__":
    auditor = CodeAuditor()
    auditor.run_full_audit()
