# 总结
不论是demo版还是episode版，single round居然起步慢，这是没想到的，demo版还直接失败(收敛慢，得分0左右停止训练了)；

demo版，img不除以255，比img/255.0的起步速度快；

demo版，img/255.0后，lr=1e-4只比原值x10倍，调到2.5e-4、5e-4都失败；

demo版，gae、norm都很重要，相比下gae又比norm更重要，它们比episode版影响更大；

demo diff版，没想象中效果好，是否与VecENv done之后返回下一局开始有关，真正的结束frame可以在info['terminal_observation']获取；


# ppo_demo
8个env同时跑，demo版，原始code从github上下载

* ppo_demo：成功，Pong-v0，基础版本，ppo_demo.custom.py 在 ppo_demo.original.py 基础上添加了summery
* ppo_demo_deterministic：成功，PongDeterministic-v0 的基础版本，得分比Pong-v0高
* ppo_demo.norm_img_lr0.0001.PongDeterministic-v0：成功，img/255.0，lr=1e-4，起步没有ppo_demo_deterministic快，最后得分差不多
* ppo_demo.norm_img_lr0.0005.PongDeterministic-v0：失败，img/255.0，lr=5e-4
* ppo_demo.norm_img_lr0.00025.PongDeterministic-v0：失败，img/255.0，lr=1e-4，img/255.0，lr=2.5e-4
* ppo_demo.no_gae_no_norm.PongDeterministic-v0：失败，img/255.0，lr=1e-4，得分-20到-10之间，计算returns时仅使用delta，不使用gae，也不normalize advantage
* ppo_demo.no_gae.PongDeterministic-v0：失败，img/255.0，lr=1e-4，得分-20到0之间
* ppo_demo.no_norm.PongDeterministic-v0：失败，img/255.0，lr=1e-4，得分0到10之间
* ppo_demo.single_round_reward.PongDeterministic-v0：失败，img/255.0，lr=1e-4，reward=-1或1时就认为done了，起步慢，得分-10到10之间
* ppo_demo.single_round.lr3e-5.K=5.PongDeterministic-v0：失败，reward=-1或1时就认为done了，lr3e-5，K=5
* ppo_demo.img255.K=1.lr=1e-4.PongDeterministic-v0：失败，不做img/255.0，K=1，lr=1e-4
* ppo_demo.diff.PongDeterministic-v0：成功，img/255.0，lr=1e-4，且前后frame做diff，起步刚开始超过了ppo_demo_deterministic一点，但又掉下去导致落后，得分差不多
* ppo_demo.same_img_feature.PongDeterministic-v0：成功，img/255.0，lr=1e-4，actor、critic共享Conv层feature，由于参数变少，起步比ppo_demo.norm_img_lr0.0001.PongDeterministic-v0快

# ppo_episode
1个env跑，每次episode done后开始训练，参考demo版的image处理、GAE(Generalized Advantage Estimates) rewards处理

* ppo_episode：成功，PongDeterministic-v0，img/255.0，lr=1e-4，未做normalize advantage，得分在10~20
* ppo_episode.single_round：半失败，PongDeterministic-v0，img/255.0，lr=1e-4，未做normalize，起步快，但逐渐落后，最终得分10分左右，比ppo_episode低
* ppo_episode.reward_as_return.PongDeterministic-v0：失败，img/255.0，lr=1e-4，将rewards*gamma后，直接作为returns
* ppo_episode.reward_as_return.norm_advatage.PongDeterministic-v0：失败，img/255.0，lr=1e-4，将rewards*gamma作为returns，normalize advantage
* ppo_episode.single_round.norm_advatage.PongDeterministic-v0：成功，起步慢，img/255.0，lr=1e-4，reward=-1或1时就认为done了，normalize advantage，得分与ppo_episode差不多，后期甚至平均略高一点点

