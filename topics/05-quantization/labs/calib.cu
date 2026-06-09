// =============================================================
// 量化第③课：校准(calibration)——把误差从 round-to-nearest 压下去
//
// RTN 的死穴：scale = maxabs/qmax，让"最大值"决定 scale。若组里有离群值，
//   scale 被撑大、正常权重只用到很少的量化级 → 误差大。
// Clip 校准：故意把 scale 取小一点(裁掉离群值，让它们饱和到 ±qmax)，
//   换来正常权重更细的分辨率 → 总误差反而更小。搜一个最优 clip 比例即可。
// 对比 per-group INT4 的 RTN vs Clip，测重建误差 + 输出误差(y=Wx)。
// =============================================================
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

static float gauss(){ float u1=(rand()+1.f)/(RAND_MAX+1.f),u2=(rand()+1.f)/(RAND_MAX+1.f);
    return sqrtf(-2*logf(u1))*cosf(6.2831853f*u2); }

// 用给定 scale 对 [w,w+n) 量化→反量化写 out，返回该组的 MSE
static double quant_group(const float* w,int n,float scale,float* out){
    double mse=0;
    for(int i=0;i<n;i++){ int q=(int)lroundf(w[i]/scale); if(q>7)q=7; if(q<-8)q=-8;
        out[i]=q*scale; double d=w[i]-out[i]; mse+=d*d; }
    return mse;
}

int main(){
    const int M=512, K=2048, G=128;          // 512 行 × 2048，每组 128
    std::vector<float> W((size_t)M*K), x(K), Wr((size_t)M*K), Wc((size_t)M*K);
    // 权重 N(0,1)，1% 概率 ×6 制造离群值
    for(size_t i=0;i<(size_t)M*K;i++){ float v=gauss(); if(rand()%100==0) v*=6.f; W[i]=v; }
    for(int k=0;k<K;k++) x[k]=gauss();

    int nG=K/G; float clipUsed=0; int clipCnt=0;
    for(int m=0;m<M;m++) for(int g=0;g<nG;g++){
        const float* wg=&W[(size_t)m*K+g*G];
        float maxabs=1e-12f; for(int j=0;j<G;j++) maxabs=fmaxf(maxabs,fabsf(wg[j]));

        // ① RTN: scale = maxabs/7
        quant_group(wg,G, maxabs/7.f, &Wr[(size_t)m*K+g*G]);

        // ② Clip: 搜 c∈[0.3,1.0]，scale=c*maxabs/7，取 MSE 最小
        double best=1e30; float bestc=1.f; std::vector<float> tmp(G);
        for(float c=1.0f;c>=0.3f;c-=0.05f){
            double mse=quant_group(wg,G, c*maxabs/7.f, tmp.data());
            if(mse<best){ best=mse; bestc=c; }
        }
        quant_group(wg,G, bestc*maxabs/7.f, &Wc[(size_t)m*K+g*G]);
        clipUsed+=bestc; clipCnt++;
    }

    auto recon_err=[&](const std::vector<float>& Wq){
        double num=0,den=0; for(size_t i=0;i<(size_t)M*K;i++){ double d=W[i]-Wq[i]; num+=d*d; den+=(double)W[i]*W[i]; }
        return sqrt(num/den)*100; };
    auto out_err=[&](const std::vector<float>& Wq){
        double num=0,den=0;
        for(int m=0;m<M;m++){ double yr=0,yq=0;
            for(int k=0;k<K;k++){ yr+=(double)W[(size_t)m*K+k]*x[k]; yq+=(double)Wq[(size_t)m*K+k]*x[k]; }
            double d=yr-yq; num+=d*d; den+=yr*yr; }
        return sqrt(num/den)*100; };

    printf("per-group INT4 (G=%d)，权重含 1%% 离群值(×6)\n\n", G);
    printf("%-16s | 重建误差 | 输出误差(Wx)\n","方法");
    printf("%-16s | %6.2f%% | %8.2f%%\n","RTN(maxabs/7)", recon_err(Wr), out_err(Wr));
    printf("%-16s | %6.2f%% | %8.2f%%\n","Clip 校准",      recon_err(Wc), out_err(Wc));
    printf("\n平均最优 clip 比例 = %.2f (即把 scale 取到 maxabs 的 %.0f%%，裁掉离群值)\n",
           clipUsed/clipCnt, clipUsed/clipCnt*100);
    return 0;
}
