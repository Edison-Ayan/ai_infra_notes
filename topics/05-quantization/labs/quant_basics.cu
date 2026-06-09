// =============================================================
// 量化第一课：scale + 量化粒度，看"离群值"如何摧毁低比特量化
//
// 对称均匀量化： scale = maxabs / qmax ; q = round(x/scale) 截断 ; deq = q*scale
// 粒度：
//   per-tensor   —— 整个矩阵 1 个 scale
//   per-channel  —— 每行(通道) 1 个 scale
//   per-group    —— 每行内每 G 个 1 个 scale (GPTQ/AWQ 用)
// 实验：少数"离群通道"幅度大 50×，看不同粒度+比特的相对误差。
// 结论预演：per-tensor INT4 会被离群值带崩(正常通道被压成 0)；per-channel/group 救回。
// =============================================================
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

// 对该段 [x, x+n) 做对称量化→反量化，写回 out。bits=4/8。
static void quant_dequant(const float* x, int n, int bits, float* out){
    float maxabs = 1e-12f;
    for(int i=0;i<n;i++) maxabs = fmaxf(maxabs, fabsf(x[i]));
    int qmax = (1<<(bits-1)) - 1;          // INT4: 7, INT8: 127
    float scale = maxabs / qmax;
    for(int i=0;i<n;i++){
        int q = (int)lroundf(x[i]/scale);
        if(q>qmax) q=qmax; if(q<-qmax-1) q=-qmax-1;   // 截断到 [-qmax-1, qmax]
        out[i] = q * scale;
    }
}

// 相对 L2 误差 ||x-deq|| / ||x||。mode: 0=全体, 1=仅正常通道, 2=仅离群通道
static float rel_err(const std::vector<float>& x, const std::vector<float>& deq,
                     int M, int N, int mode, const std::vector<int>& isOut){
    double num=0, den=0;
    for(int c=0;c<M;c++){
        if(mode==1 && isOut[c]) continue;     // 跳过离群通道
        if(mode==2 && !isOut[c]) continue;    // 跳过正常通道
        for(int j=0;j<N;j++){ int i=c*N+j; double d=x[i]-deq[i]; num+=d*d; den+=(double)x[i]*x[i]; }
    }
    return (float)sqrt(num/den);
}

static float gauss(){ // 简易高斯
    float u1=(rand()+1.0f)/(RAND_MAX+1.0f), u2=(rand()+1.0f)/(RAND_MAX+1.0f);
    return sqrtf(-2*logf(u1))*cosf(6.2831853f*u2);
}

int main(){
    const int M=128, N=1024;               // 128 通道，每通道 1024 个权重
    const int G=128;                       // per-group 的组大小
    std::vector<float> W(M*N), deq(M*N);
    std::vector<int> isOut(M,0);
    // 4 个离群通道，幅度 ×50
    for(int c=0;c<M;c++){
        float mag = (c%32==0) ? 50.0f : 1.0f;   // 每 32 通道一个离群通道
        if(mag>1.0f) isOut[c]=1;
        for(int j=0;j<N;j++) W[c*N+j] = gauss()*mag;
    }
    int nOut=0; for(int c=0;c<M;c++) nOut+=isOut[c];
    printf("权重 %dx%d，其中 %d 个离群通道(幅度×50)\n\n", M, N, nOut);

    auto run=[&](const char* name, int bits, int gran){
        // gran: 0=per-tensor, 1=per-channel, 2=per-group(G)
        if(gran==0) quant_dequant(W.data(), M*N, bits, deq.data());
        else if(gran==1) for(int c=0;c<M;c++) quant_dequant(W.data()+c*N, N, bits, deq.data()+c*N);
        else for(int c=0;c<M;c++) for(int g=0;g<N;g+=G)
                quant_dequant(W.data()+c*N+g, G, bits, deq.data()+c*N+g);
        float eAll = rel_err(W,deq,M,N,0,isOut);   // 全体
        float eNorm= rel_err(W,deq,M,N,1,isOut);   // 仅正常通道
        printf("%-22s | 全体误差 %6.2f%% | 正常通道误差 %7.2f%%\n", name, eAll*100, eNorm*100);
    };

    run("per-tensor INT8", 8, 0);
    run("per-tensor INT4", 4, 0);
    run("per-channel INT4", 4, 1);
    run("per-group INT4 (G=128)", 4, 2);

    printf("\n显存(每权重位数)：FP16=16  INT8=8  INT4=4  → INT4 比 FP16 省 4×\n");
    printf("(per-channel/group 需额外存 scale，但占比极小)\n");
    return 0;
}
