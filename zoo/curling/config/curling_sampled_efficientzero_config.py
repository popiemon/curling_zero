from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 1
n_episode = 4
evaluator_env_num = 1
continuous_action_space = True
K = 100  # num_of_sampled_actions
num_simulations = 200
update_per_collect = None
model_update_ratio = 0.25
batch_size = 128
max_env_step = int(10e6)
reanalyze_ratio = 0.
prob_random_agent = 0.5
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

curling_cont_sampled_efficientzero_config = dict(
    exp_name=
    f'data_sez_ctree/curling_cont_sampled_efficientzero_k{K}_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_rr{reanalyze_ratio}_seed0',
    env=dict(
        env_name='Curling',
        env_type='normal',
        battle_mode='self_play_mode'
        continuous=True,
        manually_discretization=False,
        prob_random_agent=prob_random_agent,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        simulation_env_name='curling'
        simulation_env_config_type='sampled_self_play'
        model=dict(
            observation_shape=(5, 96, 96),
            action_space_size=3,
            num_of_sampled_actions=K,
            num_res_blocks=1,
            num_channels=32,
            image_channel=5,
            continuous_action_space=True,
            downsample=True, 
        ),
        cuda=True,
        env_type='board_games',
        game_segment_length=200,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        # NOTE: this parameter is important for stability in bipedalwalker.
        grad_clip_value=0.5,
        # NOTE: for continuous gaussian policy, we use the policy_entropy_loss as in the original Sampled MuZero paper.
        policy_entropy_loss_weight=5e-3,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        model_update_ratio=model_update_ratio,
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
curling_cont_sampled_efficientzero_config = EasyDict(curling_cont_sampled_efficientzero_config)
main_config = curling_cont_sampled_efficientzero_config

curling_cont_sampled_efficientzero_create_config = dict(
    env=dict(
        type='Curling',
        import_names=['zoo.curling.envs.curling_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_efficientzero',
        import_names=['lzero.policy.sampled_efficientzero'],
    ),
    collector=dict(
        type='episode_muzero',
        get_train_sample=True,
        import_names=['lzero.worker.muzero_collector'],
    )
)
curling_cont_sampled_efficientzero_create_config = EasyDict(
    curling_cont_sampled_efficientzero_create_config
)
create_config = curling_cont_sampled_efficientzero_create_config

if __name__ == "__main__":
    # Users can use different train entry by specifying the entry_type.
    entry_type = "train_muzero"  # options={"train_muzero", "train_muzero_with_gym_env"}

    if entry_type == "train_muzero":
        from lzero.entry import train_muzero
    elif entry_type == "train_muzero_with_gym_env":
        """
        The ``train_muzero_with_gym_env`` entry means that the environment used in the training process is generated by wrapping the original gym environment with LightZeroEnvWrapper.
        Users can refer to lzero/envs/wrappers for more details.
        """
        from lzero.entry import train_muzero_with_gym_env as train_muzero

    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)