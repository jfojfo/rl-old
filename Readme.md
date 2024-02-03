深度学习可视化中间变量的神器Visualizer
https://www.zhihu.com/question/384519338/answer/2620414587

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

vae版：vae_ac_from_latent对应ppo_demo.vae2.py

ppo_demo.vae_recon_mean_kl_loss_c3_0.01.PongDeterministic-v0：reconstruction loss改为mean，c3、c4都调小为0.01，
起步快，比ppo_demo基础版都快，中间能训练到20，但时间长了会掉落回-20，且波动大不稳定，test reward不及基础版。

去掉Conv BatchNorm后成功！！！
之前BatchNorm失败原因可能是，model在update之外的步骤，比如env.step、算next_state得分，也调用了，会影响BatchNorm。如果限定在update步骤可以成功。

* ppo_demo.vae_latent_10_c3_0.01_c4_0.01_no_batchnorm.PongDeterministic-v0：成功，去除了Conv BatchNorm，feature是最后一层Conv，
latent=10，c3=0.01，c4=0.01，mean(recon_loss)，mean(kl_loss)
kl grad、recon grad与critic grad相差量级，且kl grad与recon grad也相差量级
* ppo_demo.vae_ac_from_latent_4_c3_1_c4_0.01_recon_sum_mean_kl_sum_mean_no_batchnorm.PongDeterministic-v0：得分10，去除了Conv BatchNorm，feature是latent输出，
latent=4，c3=1，c4=0.01，mean(sum(recon_loss,dim=(1,2,3)))，mean(sum(kl_loss,dim=1))，
调整c3 c4系数后，kl grad、recon grad与critic grad量级相当，
tensorboard中的kl_loss已经除以latent_dim（cumulative KL divergence (or “rate”, in bits per dimension)），~5.7
* ppo_demo.vae_ac_from_latent_4_c3_0.1_c4_0.001_recon_sum_mean_kl_sum_mean_no_batchnorm.PongDeterministic-v0：失败
* ppo_demo.vae_ac_from_latent_10_c3_1_c4_0.01_recon_sum_mean_kl_sum_mean_no_batchnorm.PongDeterministic-v0：成功
* ppo_demo.vae_ac_from_latent_10_c3_1_c4_0.01_recon_sum_mean_kl_sum_mean_batchnorm_only_update.PongDeterministic-v0：成功，得分略低，起步快但后续慢下来，
BatchNorm限定在update步骤
* ppo_demo.vae_ac_from_latent_10_c3_0.1_c4_0.001_recon_sum_mean_kl_sum_mean_batchnorm_only_update.PongDeterministic-v0：失败

# ppo_demo.attention

ppo_demo.attention.py

attention版，Conv后的feature组成sequence，每次step后Conv得到feature作为query，去历史sequence中查key、value得到attention

16起步比64快，后期64的test分高一些，且train得分波动小一些，两者在test时分数都有大幅下跌的情况

* ppo_demo.attention_test.PongDeterministic-v0：成功，look_back_size=16
* ppo_demo.attention_test_64.PongDeterministic-v0：成功，look_back_size=64
* ppo_demo.attention_64_pos_encode.PongDeterministic-v0：成功，look_back_size=64，加PositionalEmbedding，得分比不加pos encoding略低

前面都有mask bug（-1e-7应该改为-1e7，或者float('-inf')，-inf需要避免全部mask的case）


# ppo_demo.attention_seg_img

ppo_demo.attention_seg_img.py

attention_seg_img版，去掉Conv，只用Sequence。
将84x84 image切分成7x7块12x12的patches，每个patch展开后embed dim=144，sequence len=49
更占用GPU内存

* ppo_demo.attention_seg_img.PongDeterministic-v0：失败，一层seq，最后输出时只取144/4=36的embed dim长度，得分到-10就不涨了；
* ppo_demo.attention_cascade_4_seg_img.PongDeterministic-v0：半失败，得分0-10，波动幅度大，4层seq
  * seq1：144 embed，24 heads，288 linear；
  * seq2：AdaptiveAvgPool1d 缩减embed为72，12 heads，144 linear；
  * seq3：AdaptiveAvgPool1d 缩减embed为36，6 heads，72 linear；
  * seq4：AdaptiveAvgPool1d 缩减embed为18，3 heads，36 linear；


# ppo_demo.attention_pred

ppo_demo.attention_pred.py
ppo_demo.attention_pred_mini.py

* ppo_demo.attention_pred_16_c3_1.PongDeterministic-v0：成功，c3=1，添加一个预测分支：attn_out+action预测next_state的feature

前面都有mask bug（-1e-7应该改为-1e7，或者float('-inf')，-inf需要避免全部mask的case）

memory每次根据states计算，速度太慢

# ppo_demo.seg_img_pos

ppo_demo.seg_img_pos.py

* ppo_demo.seg_img_base.PongDeterministic-v0：成功，速度快，2h差不多，划分为12x12个7x7 patches，每个patch 3x3 conv 2次得到（16,1,1）块，再flatten得到patch_feature
* ppo_demo.seg_img_pos.PongDeterministic-v0：失败，10分左右，将2维坐标one-hot后concat到前面patch_feature（concat后dim=16+12+12），再linear回dim=16
* ppo_demo.seg_img_pos_embed.PongDeterministic-v0：不稳定，18分后又掉到10甚至-10分，用nn.Embedding(NUM_Y * NUM_X, embed_dim)直接embed位置，再加到patch_feature（参考ViT）

# attention_gtrxl_mem

ppo_demo.attention_gtrxl_mem.py

* ppo_demo.attention_gtrxl_mem.PongDeterministic-v0：15分左右，仅参考了gtrxl的mem设计，4层attn layers，look back 256，memory_concat时对 out detach(torch.cat([out.detach(), memory[i]], dim=0))
* attention_gau_look_back_256_attn_layers_4.PongDeterministic-v0：比前一个得分低一些，得分波动大，memory_concat 未对 out detach


# step reward

ppo_demo.custom.step_reward.py

* ppo_demo.custom.step_reward_0.002_norm_std.PongDeterministic-v0：20分，每step奖励0.002，advantages /= (advantages.std() + 1e-8)


# norm_std

ppo_demo.custom.norm_std.py

* ppo_demo.custom.norm_std.PongDeterministic-v0：advantages /= (advantages.std() + 1e-8)，第一次获得21分自动结束（得分曲线有一段落下），第二次20分（得分曲线正常上升)
* ppo_demo.custom.norm_std.repeat_action.PongDeterministic-v0：失败，参考论文https://arxiv.org/pdf/2305.17109.pdf

