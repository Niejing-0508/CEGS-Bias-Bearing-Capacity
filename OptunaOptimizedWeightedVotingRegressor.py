import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
import optuna
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['SimSun', "Microsoft YaHei", "SimHei"]
plt.rcParams['axes.unicode_minus'] = False

class OptunaOptimizedWeightedVotingRegressor:
    ### 修改点 1: __init__ 方法增加 manual_weights 参数 ###
    def __init__(self, n_trials=50, manual_weights=None):
        self.models = {}
        self.weights = {}
        # 如果提供了手动权重，则直接设置
        self.manual_weights = manual_weights
        self.svr_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.is_fitted = False
        self.best_params_ = {}
        self.n_trials = n_trials
        self.X_train_fit_ = None
        self.y_train_fit_ = None

    def _optimize_extra_trees(self, X_train, y_train):
        print("\n使用Optuna优化ExtraTrees超参数...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 10, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 1443,
                'n_jobs': -1
            }
            model = ExtraTreesRegressor(**params)
            scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2', n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        print(f"ExtraTrees最佳参数: {study.best_params}")
        print(f"ExtraTrees最佳分数: {study.best_value:.4f}")
        best_params = study.best_params.copy()
        best_params['random_state'] = 1443
        best_params['n_jobs'] = -1
        return ExtraTreesRegressor(**best_params)

    def _optimize_gbdt(self, X_train, y_train):
        print("\n使用Optuna优化GBDT超参数...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'random_state': 1443
            }
            model = GradientBoostingRegressor(**params)
            scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2', n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        print(f"GBDT最佳参数: {study.best_params}")
        print(f"GBDT最佳分数: {study.best_value:.4f}")
        best_params = study.best_params.copy()
        best_params['random_state'] = 1443
        return GradientBoostingRegressor(**best_params)

    def _optimize_svr(self, X_train, y_train):
        print("\n使用Optuna优化SVR超参数...")
        y_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        X_scaled = self.svr_scaler.fit_transform(X_train)

        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.1, 1000, log=True),
                'gamma': trial.suggest_float('gamma', 0.001, 1, log=True),
                'epsilon': trial.suggest_float('epsilon', 0.001, 0.5),
                'kernel': 'rbf',
                'cache_size': 1000
            }
            model = SVR(**params)
            scores = cross_val_score(model, X_scaled, y_scaled, cv=10, scoring='r2', n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        print(f"SVR最佳参数: {study.best_params}")
        print(f"SVR最佳分数: {study.best_value:.4f}")
        best_params = study.best_params.copy()
        best_params['kernel'] = 'rbf'
        best_params['cache_size'] = 1000
        return SVR(**best_params)

    def _optimize_catboost(self, X_train, y_train):
        print("\n使用Optuna优化CatBoost超参数...")

        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_strength': trial.suggest_float('random_strength', 0.1, 2),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'loss_function': 'RMSE',
                'eval_metric': 'RMSE',
                'random_state': 1443,
                'verbose': False,
                'thread_count': -1
            }
            model = CatBoostRegressor(**params)
            scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2', n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        print(f"CatBoost最佳参数: {study.best_params}")
        print(f"CatBoost最佳分数: {study.best_value:.4f}")
        best_params = study.best_params.copy()
        best_params.update({
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_state': 1443,
            'verbose': False,
            'thread_count': -1
        })
        return CatBoostRegressor(**best_params)

    def _initialize_optimized_models(self, X_train, y_train):
        print("开始Optuna超参数优化...")
        optimized_et = self._optimize_extra_trees(X_train, y_train)
        optimized_gbdt = self._optimize_gbdt(X_train, y_train)
        optimized_svr = self._optimize_svr(X_train, y_train)
        optimized_catboost = self._optimize_catboost(X_train, y_train)
        self.models = {
            'ExtraTrees': optimized_et,
            'GBDT': optimized_gbdt,
            'SVR': optimized_svr,
            'CatBoost': optimized_catboost
        }
        self.best_params_ = {
            'ExtraTrees': optimized_et.get_params(),
            'GBDT': optimized_gbdt.get_params(),
            'SVR': optimized_svr.get_params(),
            'CatBoost': optimized_catboost.get_params()
        }
    ### 修改点 2: _calculate_model_weights 方法优先使用手动权重 ###
    def _calculate_model_weights(self, X_val, y_val):
        # 如果手动权重已设置，则直接使用
        if self.manual_weights is not None:
            # 简单验证权重字典的有效性
            required_models = ['ExtraTrees', 'GBDT', 'SVR', 'CatBoost']
            if not all(model in self.manual_weights for model in required_models):
                raise ValueError(f"手动权重字典必须包含以下所有模型: {required_models}")
            if not np.isclose(sum(self.manual_weights.values()), 1.0, atol=1e-3):
                raise ValueError(f"手动权重的总和必须为 1.0")

            self.weights = self.manual_weights
            print("\n使用手动固定的模型权重:")
            for name, weight in self.weights.items():
                print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")
            return
        # 否则，执行原有的自动计算权重逻辑
        print("\n未检测到手动权重，将基于验证集性能自动计算权重...")
        performance_scores = {}
        individual_metrics = {}
        for name, model in self.models.items():
            if name == 'SVR':
                X_val_scaled = self.svr_scaler.transform(X_val)
                y_pred_scaled = model.predict(X_val_scaled)
                y_pred_original = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            else:
                y_pred_original = model.predict(X_val)
            r2 = r2_score(y_val, y_pred_original)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred_original))
            mae = mean_absolute_error(y_val, y_pred_original)
            mape = mean_absolute_percentage_error(y_val, y_pred_original)
            individual_metrics[name] = {
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            }
            if r2 < 0:
                print(f"警告: {name} 的R²为负值 ({r2:.4f})，将使用保守权重")
                performance_scores[name] = 0.01
            else:
                performance_scores[name] = max(r2, 0.05)

        print("\n各个模型在验证集上的表现:")
        for name, metrics in individual_metrics.items():
            print(f"  {name}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}")

        total_score = sum(performance_scores.values())
        self.weights = {name: score / total_score for name, score in performance_scores.items()}

        print("\n基于验证集性能自动计算的最终模型权重分配:")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")

    def fit(self, X, y, val_ratio=0.3):
        print("开始训练Optuna优化加权的投票集成模型...")
        self.X_train_fit_ = X
        self.y_train_fit_ = y

        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X, y, test_size=val_ratio, random_state=42
        )

        self._initialize_optimized_models(X_train_main, y_train_main)

        print("\n在完整训练集上重新训练优化后的模型...")
        for name, model in self.models.items():
            print(f"重新训练 {name}...")
            if name == 'SVR':
                y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
                X_scaled = self.svr_scaler.fit_transform(X)
                model.fit(X_scaled, y_scaled)
            else:
                model.fit(X, y)
        # 调用权重计算方法（该方法内部会判断是否使用手动权重）
        self._calculate_model_weights(X_val, y_val)

        self.is_fitted = True
        print("\nOptuna优化集成模型训练完成！")

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        predictions = np.zeros(len(X))
        for name, model in self.models.items():
            if name == 'SVR':
                X_scaled = self.svr_scaler.transform(X)
                pred_scaled = model.predict(X_scaled)
                pred = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            else:
                pred = model.predict(X)
            predictions += self.weights[name] * pred
        return predictions

    def predict_individual(self, X):
        individual_predictions = {}
        for name, model in self.models.items():
            if name == 'SVR':
                X_scaled = self.svr_scaler.transform(X)
                pred_scaled = model.predict(X_scaled)
                individual_predictions[name] = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            else:
                individual_predictions[name] = model.predict(X)
        return individual_predictions

    def evaluate(self, X, y):
        predictions = self.predict(X)
        metrics = {
            'MSE': mean_squared_error(y, predictions),
            'RMSE': np.sqrt(mean_squared_error(y, predictions)),
            'MAE': mean_absolute_error(y, predictions),
            'R2': r2_score(y, predictions),
            'MAPE': mean_absolute_percentage_error(y, predictions)
        }
        return metrics, predictions

    def calculate_cov_and_mean(self, X, y):
        y_pred = self.predict(X)
        mean_actual = np.mean(y)
        mean_pred = np.mean(y_pred)
        std_actual = np.std(y, ddof=1)
        std_pred = np.std(y_pred, ddof=1)
        absolute_errors = np.abs(y - y_pred)
        mean_absolute_error_val = np.mean(absolute_errors)
        std_absolute_error = np.std(absolute_errors, ddof=1)
        ratio = y_pred / y
        mean_ratio = np.mean(ratio)
        std_ratio = np.std(ratio, ddof=1)
        cov_actual = (std_actual / mean_actual) * 100 if mean_actual != 0 else 0
        cov_pred = (std_pred / mean_pred) * 100 if mean_pred != 0 else 0
        cov_absolute_error = (std_absolute_error / mean_absolute_error_val) * 100 if mean_absolute_error_val != 0 else 0
        cov_ratio = (std_ratio / mean_ratio) * 100 if mean_ratio != 0 else 0
        return {
            'MEAN_Actual': mean_actual,
            'MEAN_Predicted': mean_pred,
            'MEAN_Absolute_Error': mean_absolute_error_val,
            'MEAN_Ratio': mean_ratio,
            'STD_Actual': std_actual,
            'STD_Predicted': std_pred,
            'STD_Absolute_Error': std_absolute_error,
            'STD_Ratio': std_ratio,
            'COV_Actual(%)': cov_actual,
            'COV_Predicted(%)': cov_pred,
            'COV_Absolute_Error(%)': cov_absolute_error,
            'COV_Ratio(%)': cov_ratio
        }

    def plot_comparison(self, X_test, y_test):
        individual_preds = self.predict_individual(X_test)
        ensemble_pred = self.predict(X_test)
        model_r2 = {name: r2_score(y_test, pred) for name, pred in individual_preds.items()}
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        plt.figure(figsize=(12, 7))
        models_list = list(model_r2.keys()) + ['Optuna Ensemble']
        r2_scores = list(model_r2.values()) + [ensemble_r2]
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFB366', '#FF6B6B']
        bars = plt.bar(models_list, r2_scores, color=colors, alpha=1.0)
        plt.ylabel('R² Score', fontsize=14)
        plt.title('Performance Comparison Between Base Models and Ensemble Model', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        for bar, score in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{score:.4f}', ha='center', va='bottom', fontsize=11)
        plt.axhline(y=ensemble_r2, color='r', linestyle='--', alpha=0.7, label=f'Ensemble R² ({ensemble_r2:.4f})')
        plt.legend()
        plt.tight_layout()
        plt.savefig('CEGS模型性能对比.png', dpi=300, bbox_inches='tight')
        plt.show()
        return model_r2, ensemble_r2

    def plot_predictions_vs_actual(self, X_train, X_test, y_train, y_test):
        y_train_pred = self.predict(X_train)
        y_test_pred = self.predict(X_test)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

        plt.figure(figsize=(10, 8))
        plt.scatter(y_train, y_train_pred, color='#FF9999', s=80, marker='o', alpha=1, label=f'Train (R²={train_r2:.4f})')
        plt.scatter(y_test, y_test_pred, color='#66B2FF', s=80, marker='^', alpha=1, label=f'Test (R²={test_r2:.4f})')
        plt.xlabel('True Ultimate Bearing Capacity (kN)', fontsize=14)
        plt.ylabel('Predicted Ultimate Bearing Capacity (kN)', fontsize=14)
        plt.title("Actual vs. Predicted Values", fontsize=16)
        text_box = f"R²: {test_r2:.4f}\nMAE: {test_mae:.4f} kN\nRMSE: {test_rmse:.4f} kN\nMAPE: {test_mape:.4%}"
        plt.text(50, 2500, text_box, fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.plot([0, 3300], [0, 3300], "--k")
        plt.plot([0, 3300], [0 * 1.15, 3300 * 1.15], "--b", alpha=0.5)
        plt.plot([0, 3300], [0 * 0.85, 3300 * 0.85], "--b", alpha=0.5, label='±15% Error Band')
        plt.xlim(0, 3300)
        plt.ylim(0, 3300)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('CEGS Regression.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\n训练集指标:")
        print(f"R²: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}")
        print(f"\n测试集指标:")
        print(f"R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}")

        train_cov_metrics = self.calculate_cov_and_mean(X_train, y_train)
        test_cov_metrics = self.calculate_cov_and_mean(X_test, y_test)

        print(f"\n训练集COV和MEAN统计:")
        for key, value in train_cov_metrics.items():
            print(f"  {key}: {value:.4f}")
        print(f"\n测试集COV和MEAN统计:")
        for key, value in test_cov_metrics.items():
            print(f"  {key}: {value:.4f}")

        return y_train_pred, y_test_pred, train_cov_metrics, test_cov_metrics