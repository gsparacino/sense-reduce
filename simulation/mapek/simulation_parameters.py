import datetime

from base.learning_strategy import NoUpdateStrategy, RetrainStrategy
from base.portfolio_adaptation_goal import DeployOnceAdaptationGoal, FixedIntervalAdaptationGoal

portfolios = {
    # Vienna Portfolios
    "vienna_2019_2019_simple_dense": {
        "models": ["zamg_vienna_2019_2019_simple_dense"],
        "legacy_mode": True,
        "training_df": "vienna_2019_2019.pickle",
        "initial_df": "vienna_2019_2019.pickle",
    },
    "vienna_2019_2019_simple_lstm": {
        "models": ["zamg_vienna_2019_2019_simple_lstm"],
        "legacy_mode": True,
        "training_df": "vienna_2019_2019.pickle",
        "initial_df": "vienna_2019_2019.pickle",
    },
    "vienna_2019_2019_conv_lstm": {
        "models": ["zamg_vienna_2019_2019_conv_lstm"],
        "legacy_mode": True,
        "training_df": "vienna_2019_2019.pickle",
        "initial_df": "vienna_2019_2019.pickle",
    },
    "vienna_2019_2019_mixed_arch": {
        "models": [
            "zamg_vienna_2019_2019_simple_dense",
            "zamg_vienna_2019_2019_conv_lstm",
            "zamg_vienna_2019_2019_simple_lstm",
        ],
        "legacy_mode": False,
        "training_df": "vienna_2019_2019.pickle",
        "initial_df": "vienna_2019_2019.pickle",
    },
    "vienna_2017_2019_simple_dense": {
        "models": ["vienna_2017_2019_simple_dense"],
        "legacy_mode": True,
        "training_df": "vienna_2017_2019.pickle",
        "initial_df": "vienna_2019_2019.pickle",
    },
    "vienna_2017_2019_simple_lstm": {
        "models": ["vienna_2017_2019_simple_lstm"],
        "legacy_mode": True,
        "training_df": "vienna_2017_2019.pickle",
        "initial_df": "vienna_2019_2019.pickle",
    },
    "vienna_2017_2019_conv_lstm": {
        "models": ["vienna_2017_2019_conv_lstm"],
        "legacy_mode": True,
        "training_df": "vienna_2017_2019.pickle",
        "initial_df": "vienna_2019_2019.pickle",
    },
    "vienna_2017_2019_mixed_arch": {
        "models": [
            "vienna_2017_2019_simple_dense",
            "vienna_2017_2019_conv_lstm",
            "vienna_2017_2019_simple_lstm",
        ],
        "legacy_mode": False,
        "training_df": "vienna_2017_2019.pickle",
        "initial_df": "vienna_2019_2019.pickle",
    },
    "vienna_2010_2019_simple_dense": {
        "models": ["zamg_vienna_2010_2019_simple_dense"],
        "legacy_mode": True,
        "training_df": "vienna_2010_2019.pickle",
        "initial_df": "vienna_2019_2019.pickle",
    },
    "vienna_2010_2019_simple_lstm": {
        "models": ["zamg_vienna_2010_2019_simple_lstm"],
        "legacy_mode": True,
        "training_df": "vienna_2010_2019.pickle",
        "initial_df": "vienna_2019_2019.pickle",
    },
    "vienna_2010_2019_conv_lstm": {
        "models": ["zamg_vienna_2010_2019_conv_lstm"],
        "legacy_mode": True,
        "training_df": "vienna_2010_2019.pickle",
        "initial_df": "vienna_2019_2019.pickle",
    },
    "vienna_2010_2019_mixed_arch": {
        "models": [
            "zamg_vienna_2010_2019_simple_dense",
            "zamg_vienna_2010_2019_conv_lstm",
            "zamg_vienna_2010_2019_simple_lstm",
        ],
        "legacy_mode": False,
        "training_df": "vienna_2010_2019.pickle",
        "initial_df": "vienna_2019_2019.pickle",
    },
    "vienna_2017_2019_simple_dense_seasons": {
        "models": [
            "vienna_winter_2017_2019_simple_dense",
            "vienna_spring_2017_2019_simple_dense",
            "vienna_summer_2017_2019_simple_dense",
            "vienna_autumn_2017_2019_simple_dense",
        ],
        "legacy_mode": False,
        "training_df": "vienna_2017_2019.pickle",
        "initial_df": "vienna_2019_2019.pickle",
    },
    "vienna_2017_2019_simple_lstm_seasons": {
        "models": [
            "vienna_winter_2017_2019_simple_lstm",
            "vienna_spring_2017_2019_simple_lstm",
            "vienna_summer_2017_2019_simple_lstm",
            "vienna_autumn_2017_2019_simple_lstm",
        ],
        "legacy_mode": False,
        "training_df": "vienna_2017_2019.pickle",
        "initial_df": "vienna_2019_2019.pickle",
    },
    "vienna_2017_2019_conv_lstm_seasons": {
        "models": [
            "vienna_winter_2017_2019_conv_lstm",
            "vienna_spring_2017_2019_conv_lstm",
            "vienna_summer_2017_2019_conv_lstm",
            "vienna_autumn_2017_2019_conv_lstm",
        ],
        "legacy_mode": False,
        "training_df": "vienna_2017_2019.pickle",
        "initial_df": "vienna_2019_2019.pickle",
    },

    # Linz Portfolios
    "linz_2010_2019_simple_dense": {
        "models": ["zamg_linz_2010_2019_simple_dense"],
        "legacy_mode": True,
        "training_df": "linz_2010_2019.pickle",
        "initial_df": "linz_2019_2019.pickle",
    },
    "linz_2010_2019_simple_lstm": {
        "models": ["zamg_linz_2010_2019_simple_lstm"],
        "legacy_mode": True,
        "training_df": "linz_2010_2019.pickle",
        "initial_df": "linz_2019_2019.pickle",
    },
    "linz_2010_2019_conv_lstm": {
        "models": ["zamg_linz_2010_2019_conv_lstm"],
        "legacy_mode": True,
        "training_df": "linz_2010_2019.pickle",
        "initial_df": "linz_2019_2019.pickle",
    },
    "linz_2010_2019_mixed_arch": {
        "models": [
            "zamg_linz_2010_2019_simple_dense",
            "zamg_linz_2010_2019_simple_lstm",
            "zamg_linz_2010_2019_conv_lstm"
        ],
        "legacy_mode": True,
        "training_df": "linz_2010_2019.pickle",
        "initial_df": "linz_2019_2019.pickle",
    },
    "linz_2019_2019_simple_dense": {
        "models": ["linz_2019_2019_simple_dense"],
        "legacy_mode": True,
        "training_df": "linz_2019_2019.pickle",
        "initial_df": "linz_2019_2019.pickle",
    },
    "linz_2019_2019_simple_lstm": {
        "models": ["linz_2019_2019_simple_lstm"],
        "legacy_mode": True,
        "training_df": "linz_2019_2019.pickle",
        "initial_df": "linz_2019_2019.pickle",
    },
    "linz_2019_2019_conv_lstm": {
        "models": ["linz_2019_2019_conv_lstm"],
        "legacy_mode": True,
        "training_df": "linz_2019_2019.pickle",
        "initial_df": "linz_2019_2019.pickle",
    },
    "linz_2019_2019_mixed_arch": {
        "models": [
            "linz_2019_2019_simple_dense",
            "linz_2019_2019_simple_lstm",
            "linz_2019_2019_conv_lstm"
        ],
        "legacy_mode": True,
        "training_df": "linz_2019_2019.pickle",
        "initial_df": "linz_2019_2019.pickle",
    },
}

# (Violations, Hours)
violation_rates = [
    (1, 1),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),
    (3, 3)
]

# (LearningStrategy, PortfolioAdaptationGoal)
strategies = {
    "static": (NoUpdateStrategy, DeployOnceAdaptationGoal("deploy_once")),
    "retrain_short": (
        RetrainStrategy, FixedIntervalAdaptationGoal("fixed_interval", datetime.timedelta(seconds=7257600))
    ),
    "retrain_long": (
        RetrainStrategy, FixedIntervalAdaptationGoal("fixed_interval", datetime.timedelta(seconds=15724800))
    ),
}
