.
├── README.markdown
├── build
│   ├── bdist.macosx-13-arm64
│   └── lib
│       └── loyalwingmen
│           ├── __init__.py
│           └── apps
│               ├── __init__.py
│               ├── baysian_optimizer_app.py
│               ├── env_interactive_app.py
│               ├── executor.py
│               ├── model_executor.py
│               ├── model_executor_simplified_app.py
│               ├── model_trainer2.py
│               ├── model_trainer_app.py
│               └── trainer.py
├── dist
│   └── loyal_wingmen-0.0.1-py3.10.egg
├── loyalwingmen
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-38.pyc
│   ├── apps
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   └── __init__.cpython-38.pyc
│   │   ├── baysian_optimizer_app.py
│   │   ├── env_interactive_app.py
│   │   ├── executor.py
│   │   ├── ml
│   │   │   ├── __pycache__
│   │   │   │   ├── directory_manager.cpython-38.pyc
│   │   │   │   └── pipeline.cpython-38.pyc
│   │   │   ├── directory_manager.py
│   │   │   └── pipeline.py
│   │   ├── model_executor.py
│   │   ├── model_executor_simplified_app.py
│   │   ├── model_trainer2.py
│   │   ├── model_trainer_app.py
│   │   └── trainer.py
│   ├── assets
│   │   ├── architrave.urdf
│   │   ├── cf2.dae
│   │   ├── cf2p.urdf
│   │   ├── cf2x.urdf
│   │   ├── example_trace.pkl
│   │   ├── hb.urdf
│   │   └── quad.obj
│   ├── loyalwingmen.egg-info
│   │   ├── PKG-INFO
│   │   ├── SOURCES.txt
│   │   ├── dependency_links.txt
│   │   └── top_level.txt
│   ├── modules
│   │   ├── enums
│   │   │   ├── __pycache__
│   │   │   │   └── entity_types.cpython-38.pyc
│   │   │   └── entity_types.py
│   │   ├── environments
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── demo_env.cpython-311.pyc
│   │   │   │   ├── demo_env.cpython-38.pyc
│   │   │   │   ├── drone_and_cube_env.cpython-38.pyc
│   │   │   │   ├── environment_models.cpython-38.pyc
│   │   │   │   └── level1_environment.cpython-38.pyc
│   │   │   ├── dataclasses
│   │   │   │   ├── __pycache__
│   │   │   │   │   └── environment_parameters.cpython-38.pyc
│   │   │   │   └── environment_parameters.py
│   │   │   ├── helpers
│   │   │   │   ├── __pycache__
│   │   │   │   │   └── normalization.cpython-38.pyc
│   │   │   │   └── normalization.py
│   │   │   ├── level1_environment.py
│   │   │   ├── olds
│   │   │   │   ├── drone_chase_env.py
│   │   │   │   ├── randomized_drone_chase_env.py
│   │   │   │   ├── randomized_drone_chase_env_action_fixed.py
│   │   │   │   └── simplified_env.py
│   │   │   └── simulations
│   │   │       ├── __pycache__
│   │   │       │   └── level1_simulation.cpython-38.pyc
│   │   │       └── level1_simulation.py
│   │   ├── events
│   │   │   ├── __pycache__
│   │   │   │   ├── broker.cpython-38.pyc
│   │   │   │   └── message_hub.cpython-38.pyc
│   │   │   ├── broker.py
│   │   │   └── message_hub.py
│   │   ├── factories
│   │   │   ├── __pycache__
│   │   │   │   ├── accessories_factory.cpython-38.pyc
│   │   │   │   ├── callback_factory.cpython-38.pyc
│   │   │   │   ├── drone_factory.cpython-38.pyc
│   │   │   │   ├── factory_models.cpython-38.pyc
│   │   │   │   ├── loiteringmunition_factory.cpython-38.pyc
│   │   │   │   ├── loyalwingman_factory.cpython-38.pyc
│   │   │   │   └── obstacle_factory.cpython-38.pyc
│   │   │   └── callback_factory.py
│   │   ├── policies
│   │   │   ├── policy.py
│   │   │   └── policy_factory.py
│   │   ├── quadcoters
│   │   │   ├── __pycache__
│   │   │   │   ├── loiteringmunition.cpython-38.pyc
│   │   │   │   ├── loyalwingman.cpython-38.pyc
│   │   │   │   └── quadcopter_factory.cpython-38.pyc
│   │   │   ├── base
│   │   │   │   ├── __pycache__
│   │   │   │   │   └── quadcopter.cpython-38.pyc
│   │   │   │   └── quadcopter.py
│   │   │   ├── components
│   │   │   │   ├── actuators
│   │   │   │   │   ├── __pycache__
│   │   │   │   │   │   ├── actuator_interface.cpython-38.pyc
│   │   │   │   │   │   └── propulsion.cpython-38.pyc
│   │   │   │   │   ├── actuator_interface.py
│   │   │   │   │   └── propulsion.py
│   │   │   │   ├── controllers
│   │   │   │   │   ├── BaseControl.py
│   │   │   │   │   ├── DSLPIDControl.py
│   │   │   │   │   ├── SimplePIDControl.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── __pycache__
│   │   │   │   │       ├── BaseControl.cpython-38.pyc
│   │   │   │   │       ├── DSLPIDControl.cpython-38.pyc
│   │   │   │   │       └── __init__.cpython-38.pyc
│   │   │   │   ├── dataclasses
│   │   │   │   │   ├── __pycache__
│   │   │   │   │   │   ├── flight_state.cpython-38.pyc
│   │   │   │   │   │   ├── operational_constraints.cpython-38.pyc
│   │   │   │   │   │   └── quadcopter_specs.cpython-38.pyc
│   │   │   │   │   ├── flight_state.py
│   │   │   │   │   ├── operational_constraints.py
│   │   │   │   │   └── quadcopter_specs.py
│   │   │   │   └── sensors
│   │   │   │       ├── __pycache__
│   │   │   │       │   ├── imu.cpython-38.pyc
│   │   │   │       │   ├── lidar.cpython-38.pyc
│   │   │   │       │   └── sensor_interface.cpython-38.pyc
│   │   │   │       ├── imu.py
│   │   │   │       ├── lidar.py
│   │   │   │       └── sensor_interface.py
│   │   │   ├── loiteringmunition.py
│   │   │   ├── loyalwingman.py
│   │   │   └── quadcopter_factory.py
│   │   └── utils
│   │       ├── Logger.py
│   │       ├── __pycache__
│   │       │   ├── Logger.cpython-38.pyc
│   │       │   ├── enums.cpython-38.pyc
│   │       │   ├── keyboard_listener.cpython-311.pyc
│   │       │   ├── keyboard_listener.cpython-38.pyc
│   │       │   └── utils.cpython-38.pyc
│   │       ├── displaytext.py
│   │       ├── enums.py
│   │       ├── keyboard_listener.py
│   │       ├── keyboard_test.py
│   │       └── utils.py
│   ├── outputs
│   │   └── baysian_optimizer_app.py
│   │       ├── ULTIMO_RANDOM_no_physics_1.00M_steps_lidar_range_20m_32_20s
│   │       │   └── logs
│   │       │       ├── h[128, 256, 128]-f15-lr1e-05
│   │       │       │   └── events.out.tfevents.1693140502.Davi-PC.37336.5
│   │       │       └── h[256, 512, 256]-f5-lr0.01
│   │       │           └── events.out.tfevents.1693141813.Davi-PC.37336.6
│   │       └── ULTIMO_no_physics_2.00M_steps_lidar_range_100m_16_20s
│   │           └── logs
│   │               └── h[256, 256, 256]-f15-lr1e-07
│   │                   └── events.out.tfevents.1693099628.Davi-PC.27896.5
│   └── tests
│       ├── reward_conditions.py
│       ├── reward_test.py
│       ├── test_directory_manager.py
│       └── test_pipeline.py
├── loyalwingmen.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   ├── requires.txt
│   ├── top_level.txt
│   └── zip-safe
├── project_structure.py
├── pyproject.toml
├── requirements-mac.txt
├── requirements-windows.txt
├── setup.cfg
├── setup.py
└── tutorial for virtual env.rtf

58 directories, 136 files
