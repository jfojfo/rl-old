# 总结
不论是demo版还是episode版，single round(reward=-1或1时就认为done了)居然起步慢，这是没想到的，
demo版还直接失败(收敛慢，得分0左右停止训练了)，single round结束时己方球拍实际上是可以上下移动的，不会重置到中间位置，所以重置为done不合理；

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

# ppo_demo.vae

vae版

ppo_demo.vae_recon_mean_kl_loss_c3_0.01.PongDeterministic-v0：reconstruction loss改为mean，c3、c4都调小为0.01，
起步快，比ppo_demo基础版都快，中间能训练到20，但时间长了会掉落回-20，且波动大不稳定，test reward不及基础版。

去掉Conv BatchNorm后成功！！！

* ppo_demo.vae_latent_10_c3_0.01_c4_0.01_no_batchnorm.PongDeterministic-v0：成功，去除了Conv BatchNorm，mean recon_loss，mean kl_loss，c3=0.01，c4=0.01

# ppo_demo.attention

attention版，Conv后的feature组成sequence，每次step后Conv得到feature作为query，去历史sequence中查key、value得到attention

16起步比64快，后期64的test分高一些，且train得分波动小一些，两者在test时分数都有大幅下跌的情况

* ppo_demo.attention_test.PongDeterministic-v0：成功，look_back_size=16
* ppo_demo.attention_test_64.PongDeterministic-v0：成功，look_back_size=64
* ppo_demo.attention_64_pos_encode.PongDeterministic-v0：成功，look_back_size=64，加PositionalEmbedding，得分比不加pos encoding略低

# ppo_demo.attention_seg_img

attention_seg_img版，去掉Conv，只用Sequence。
将84x84 image切分成7x7块12x12的patches，每个patch展开后embed dim=144，sequence len=49
更占用GPU内存

* ppo_demo.attention_seg_img.PongDeterministic-v0：失败，一层seq，最后输出时只取144/4=36的embed dim长度，得分到-10就不涨了；
* ppo_demo.attention_cascade_4_seg_img.PongDeterministic-v0：半失败，得分0-10，波动幅度大，4层seq
  * seq1：144 embed，24 heads，288 linear；
  * seq2：AdaptiveAvgPool1d 缩减embed为72，12 heads，144 linear；
  * seq3：AdaptiveAvgPool1d 缩减embed为36，6 heads，72 linear；
  * seq4：AdaptiveAvgPool1d 缩减embed为18，3 heads，36 linear；



