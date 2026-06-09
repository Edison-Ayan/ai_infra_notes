// =============================================================
// 量化第④课：简化版 AWQ——用激活信息保护"重要权重"
//
// 核心：y=Wx=Σ W[:,k]·x_k，激活 x_k 大的通道权重更"重要"(误差被放大)。
// AWQ：恒等变形 Wx = (W·diag(s))·(diag(1/s)·x)，量化放大后的 W·diag(s)，
//      对重要通道(大激活)取 s_k>1 → 其权重占更多量化级 → 相对误差↓(被保护)。
//      s_k = (a_k)^α，a_k=通道平均激活幅度，α 网格搜索(最小化输出误差)。
// 场景：少数通道激活特别大(activation outlier)，看 AWQ vs RTN/Clip 的输出误差。
// =============================================================
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
using std::vector;

static float gauss(){ float u1=(rand()+1.f)/(RAND_MAX+1.f),u2=(rand()+1.f)/(RAND_MAX+1.f);
    return sqrtf(-2*logf(u1))*cosf(6.2831853f*u2); }

// per-group INT4 量化：列预缩放 s[K]，可选 clip 搜索；写回反量化(已除回 s)的 Wq
static void quantize(const vector<float>& W,int M,int K,int G,
                     const vector<float>& s,bool doClip,vector<float>& Wq){
    for(int m=0;m<M;m++) for(int g=0;g<K/G;g++){
        float maxabs=1e-12f;
        for(int j=0;j<G;j++){ int k=g*G+j; maxabs=fmaxf(maxabs,fabsf(W[(size_t)m*K+k]*s[k])); }
        float sc=maxabs/7.f;
        if(doClip){ double best=1e30; float bc=1.f;
            for(float c=1.f;c>=0.3f;c-=0.05f){ double mse=0;
                for(int j=0;j<G;j++){ int k=g*G+j; float v=W[(size_t)m*K+k]*s[k];
                    int q=(int)lroundf(v/(c*maxabs/7.f)); if(q>7)q=7; if(q<-8)q=-8;
                    double d=v-q*(c*maxabs/7.f); mse+=d*d; }
                if(mse<best){best=mse;bc=c;} }
            sc=bc*maxabs/7.f; }
        for(int j=0;j<G;j++){ int k=g*G+j; float v=W[(size_t)m*K+k]*s[k];
            int q=(int)lroundf(v/sc); if(q>7)q=7; if(q<-8)q=-8;
            Wq[(size_t)m*K+k]=q*sc/s[k]; }   // 反量化后除回 s
    }
}

// 输出相对误差 ‖WX-WqX‖/‖WX‖，X 为 [K×S]
static double out_err(const vector<float>& W,const vector<float>& Wq,int M,int K,
                      const vector<float>& X,int S){
    double num=0,den=0;
    for(int m=0;m<M;m++) for(int sIdx=0;sIdx<S;sIdx++){
        double yr=0,yq=0;
        for(int k=0;k<K;k++){ yr+=(double)W[(size_t)m*K+k]*X[(size_t)k*S+sIdx];
                              yq+=(double)Wq[(size_t)m*K+k]*X[(size_t)k*S+sIdx]; }
        double d=yr-yq; num+=d*d; den+=yr*yr; }
    return sqrt(num/den)*100;
}

int main(){
    const int M=256,K=1024,G=128,Sc=32,St=32;   // 校准/测试各 32 样本
    vector<float> W((size_t)M*K), mag(K), Xc((size_t)K*Sc), Xt((size_t)K*St), Wq((size_t)M*K);
    // 通道激活幅度：每 16 个通道有 1 个"离群通道"幅度×10
    for(int k=0;k<K;k++) mag[k]=(k%16==0)?10.f:1.f;
    for(size_t i=0;i<(size_t)M*K;i++) W[i]=gauss();             // 权重正常 N(0,1)
    for(int k=0;k<K;k++) for(int s=0;s<Sc;s++) Xc[(size_t)k*Sc+s]=gauss()*mag[k];
    for(int k=0;k<K;k++) for(int s=0;s<St;s++) Xt[(size_t)k*St+s]=gauss()*mag[k];

    // 校准统计：a_k = 平均激活幅度
    vector<float> a(K,0);
    for(int k=0;k<K;k++){ double s=0; for(int j=0;j<Sc;j++) s+=fabsf(Xc[(size_t)k*Sc+j]); a[k]=s/Sc; }
    double glog=0; for(int k=0;k<K;k++) glog+=log(a[k]+1e-9); double geo=exp(glog/K);

    vector<float> one(K,1.f);
    quantize(W,M,K,G,one,false,Wq); double eRTN = out_err(W,Wq,M,K,Xt,St);
    quantize(W,M,K,G,one,true ,Wq); double eClip= out_err(W,Wq,M,K,Xt,St);

    // AWQ：网格搜 α，用校准集选最优
    double bestCal=1e30; float bestA=0; vector<float> s(K);
    printf("AWQ α 搜索(校准集输出误差):\n");
    for(float al=0.f;al<=1.0001f;al+=0.1f){
        for(int k=0;k<K;k++) s[k]=powf(a[k]/geo, al);
        quantize(W,M,K,G,s,false,Wq);
        double ec=out_err(W,Wq,M,K,Xc,Sc);
        printf("  α=%.1f → %.2f%%\n", al, ec);
        if(ec<bestCal){bestCal=ec;bestA=al;}
    }
    for(int k=0;k<K;k++) s[k]=powf(a[k]/geo,bestA);
    quantize(W,M,K,G,s,false,Wq); double eAWQ = out_err(W,Wq,M,K,Xt,St);

    printf("\n权重 %dx%d, per-group INT4, 每16通道1个激活离群(×10)\n", M,K);
    printf("%-18s | 测试集输出误差\n","方法");
    printf("%-18s | %.2f%%\n","RTN", eRTN);
    printf("%-18s | %.2f%%\n","Clip", eClip);
    printf("%-18s | %.2f%%  (最优 α=%.1f)\n","AWQ(激活感知)", eAWQ, bestA);
    return 0;
}
