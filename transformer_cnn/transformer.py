import numpy as np

# 论文3.4 
def softmax(x):
    exp_x = np.exp(x - np.max(x,axis=1,keepdims=True))  #了避免指数计算中的溢出问题，将输入向量减去最大值
    return exp_x / np.sum(exp_x,axis=1,keepdims=True)   #keepdims=True 保证结果的维度与原数组一致

# 论文3.2.1 这个计算自注意力机制包括有掩码和没有掩码的情况
def self_attention(Q,K,V,mask):
    d_k=Q.shape[-1]     
    scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)       # 相当于论文中attention函数表达式softmax函数的自变量

    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)  # 应用 mask  在翻译任务中，解码器部分不能看到未来的词汇，因此需要屏蔽未来的词汇。在这种情况下，如果 mask 中的某些位置为 0，那么将这些位置的 scores 设置为一个非常小的值（如 -1e9），从而在 softmax 中得到接近于 0 的注意力权重，使这些位置对最终结果没有影响。

    attention_weights = softmax(scores)     # 计算注意力权重，结合论文中的公式
    output = np.matmul(attention_weights, V)    #计算注意力输出
    return output, attention_weights

# 论文3.2.2 同样可以包括有无掩码的情况
class MultiHeadAttention:
    def __init__(self,d_model,heads):
        self.d_model=d_model
        self.heads=heads
        self.d_k=self.d_model // self.heads
        assert(self.d_k * self.heads == self.d_model),'EMBED SIZE SHOULD BE DIV BY HEADS'

        # W_q, W_k, W_v：用于将输入分别投影到查询（Query）、键（Key）和值（Value）向量的权重矩阵。它们的维度都是 (d_model, d_model)，用于线性变换输入数据。
        #这些权重都是随机初始化，之后的优化过程会更新权重
        self.W_q = np.random.rand(d_model, d_model)  # 查询向量的权重矩阵
        self.W_k = np.random.rand(d_model, d_model)  # 键向量的权重矩阵
        self.W_v = np.random.rand(d_model, d_model)  # 值向量的权重矩阵
        self.W_o = np.random.rand(d_model, d_model)  # 输出的线性变换权重矩阵，用于最终整合多头自注意力输出的权重矩阵

    def split_heads(self,x,batch_size):
        # 这个函数将输入 x 拆分成多个头。它的输入是一个形状为 (batch_size, seq_len, d_model) 的张量，表示每个输入序列的特征。
        # 它将输入重新调整成 (batch_size, seq_len, num_heads, d_k) 的形状。这里的 num_heads 是注意力头的数量，d_k 是每个头的维度。
        # 然后通过 np.transpose 将维度调整为 (batch_size, num_heads, seq_len, d_k)，这样我们就可以独立地在每个头上进行自注意力计算。
        x = x.reshape(batch_size, -1, self.heads, self.d_k)
        return np.transpose(x, (0, 2, 1, 3))  # 调整维度以适应多头
    
    def forward(self,Q,K,V,mask):
        batch_size = Q.shape[0]
        # 对输入的查询、键、值向量进行线性变换，将它们投影到新的维度空间
        Q = np.matmul(Q, self.W_q)
        K = np.matmul(K, self.W_k)
        V = np.matmul(V, self.W_v)

        # 将线性变换后的 Q, K, V 通过 split_heads 函数拆分为多个头。新的形状变为 (batch_size, num_heads, seq_len, d_k)，允许每个注意力头独立计算。
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # 计算自注意力
        attention_output, _ = self_attention(Q, K, V, mask)     #只关注第一个返回值

        # 拼接注意力头输出
        attention_output = np.transpose(attention_output, (0, 2, 1, 3)).reshape(batch_size, -1, self.d_model)

        #线性变换输出
        output = np.matmul(attention_output, self.W_o)

        return output

# 论文3.3 前馈神经网络
class Feed_Forward:
    def __init__(self,d_model,d_ff):
        # 初始化两层网络权重
        self.W1=np.random.rand(d_model,d_ff)
        self.W2=np.random.rand(d_ff,d_model)

    def forward(self,x):
        #FFN(x) = max(0, xW1 + b1)W2 + b2
        FFN=np.matmul(np.maximum(0,np.matmul(x,self.W1)),self.W2)
        return FFN
    
# 层归一化 Add&Norm
class LayerNorm:
    def __init__(self,d_model,epsilon=1e-6):
        # 初始化学习参数γ和偏移量β
        self.gamma=np.ones(d_model)
        self.beta=np.zeros(d_model)
        self.epsilon=epsilon
    
    def forward(self,x):
        # norm = (x-mean)/(variance+epsilon)^(1/2)
        mean = np.mean(x,axis=-1,keepdims=True)
        variance = np.var(x,axis=-1,keepdims=True)
        norm_x = (x-mean)/(np.sqrt(variance+self.epsilon))
        return self.gamma*norm_x + self.beta

# encoder layer
class EncoderLayer:
    def __init__(self,d_model,heads,d_ff):
        # 初始化编码层的四个结构：多头自注意力机制、第一个归一化层、前馈神经网络、第二个归一化层
        self.MultiHeadAttention = MultiHeadAttention(d_model,heads)
        self.Norm1=LayerNorm(d_model,epsilon=1e-6)
        self.FeedForward = Feed_Forward(d_model,d_ff)
        self.Norm2=LayerNorm(d_model,epsilon=1e-6)

    # encoder层没有掩码
    def forward(self,x,mask=None):
        # 输入x逐次进入四个层，得到最终输出
        output1 = self.MultiHeadAttention.forward(x,x,x,mask)
        # 层归一化的输出：LayerNorm(x + Sublayer(x))
        output1 = self.Norm1.forward(x+output1)

        output2 = self.FeedForward.forward(output1)
        # 层归一化的输出：LayerNorm(x + Sublayer(x)), 这里的x相当于第一个层归一化的输出output1
        output2 = self.Norm2.forward(output1+output2)

        return output2

# encoder layer
class DecoderLayer:
    def __init__(self,d_model,heads,d_ff):
        # 初始化编码层的6个结构：掩码多头自注意力机制、第一个归一化层、多头自注意力机制、第二个归一化层、前馈神经网络、第三个归一化层
        self.MaskedMultiHeadAttention = MultiHeadAttention(d_model,heads)
        self.Norm1=LayerNorm(d_model,epsilon=1e-6)
        self.MultiHeadAttention = MultiHeadAttention(d_model,heads)
        self.Norm2=LayerNorm(d_model,epsilon=1e-6)
        self.FeedForward = Feed_Forward(d_model,d_ff)
        self.Norm3=LayerNorm(d_model,epsilon=1e-6)

    # decoder层有掩码
    def forward(self,x,encoder_output,mask_encoder,mask_decoder):
        # 输入x逐次进入6个层，得到最终输出
        output1 = self.MaskedMultiHeadAttention.forward(x,x,x,mask_decoder)
        # 层归一化的输出：LayerNorm(x + Sublayer(x))
        output1 = self.Norm1.forward(x+output1)

        output2 = self.MultiHeadAttention.forward(output1,encoder_output,encoder_output,mask_encoder)
        # 层归一化的输出：LayerNorm(x + Sublayer(x)), 这里的x相当于第一个层归一化的输出output1
        output2 = self.Norm2.forward(output1+output2)

        output3 = self.FeedForward.forward(output2)
        # 层归一化的输出：LayerNorm(x + Sublayer(x)), 这里的x相当于第一个层归一化的输出output2
        output3 = self.Norm2.forward(output2+output3)

        return output3
    
# encoder 由n个encoder layer堆叠而成，在论文中n=6
class Encoder:
    def __init__(self,n,d_model,heads,d_ff):
        self.encoderlayers = [EncoderLayer(d_model,heads,d_ff) for _ in range(n)]

    def forward(self,x,mask=None):
        for encoderlayer in self.encoderlayers:
            x = encoderlayer.forward(x,mask)
        return x
    
# decoder 由n个decoder layer堆叠而成，在论文中n=6
class Decoder:
    def __init__(self,n,d_model,heads,d_ff):
        self.decoderlayers = [DecoderLayer(d_model,heads,d_ff) for _ in range(n)]

    def forward(self,x,encoder_output,mask_encoder,mask_decoder):
        for decoderlayer in self.decoderlayers:
            x = decoderlayer.forward(x,encoder_output,mask_encoder,mask_decoder)
        return x
    
# 完整的transformer
class Transformer:
    def __init__(self,n_encoderlayers,n_decoderlayers,d_model,heads,d_ff):
        self.encoder = Encoder(n_encoderlayers,d_model,heads,d_ff)
        self.decoder = Decoder(n_decoderlayers,d_model,heads,d_ff)
    
    def forward(self,x_encoder,x_decoder,mask_encoder,mask_decoder):
        encoder_output = self.encoder.forward(x_encoder,mask_encoder)
        decoder_output = self.decoder.forward(x_decoder,encoder_output,mask_encoder,mask_decoder)
        return decoder_output
    
# 根据论文定义超参数
n_encoder = 6
n_decoder = 6
d_model = 512
d_ff = 2048
heads = 8
