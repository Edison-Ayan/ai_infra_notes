// =============================================================
// 推理第③课：PagedAttention —— 像 OS 分页一样管 KV cache 显存
//
// 痛点：传统给每个请求预留一整块"连续"显存，大小按"可能的最大长度"。
//   但大多数请求很短 → 预留的大半空着 → 巨量内部碎片，利用率 ~20-40%。
// 解法(vLLM)：KV cache 切成固定小块(page，如 16 token)，请求按需分配、可非连续。
//   只在"最后一页"有零头浪费 → 利用率 ~95%+ → 同显存塞更多请求 → batch 更大 → 吞吐更高。
// 模拟：同一显存预算 + 同一批请求(长度长尾分布)，对比两种分配。
// =============================================================
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(){
    const int budget = 16384;   // KV cache 总容量(token 槽)
    const int maxseq = 2048;    // 每个请求"可能"的最大长度(传统按这个预留)
    const int page   = 16;      // 分页大小
    const int N      = 300;     // 候选请求

    // 真实长尾：90% 短请求(64~400)，10% 长请求(可达 ~1800)
    std::vector<int> len(N);
    long sumlen=0,mn=1<<30,mx=0;
    for(int i=0;i<N;i++){
        int L = (rand()%10<9) ? 64+rand()%340 : 400+rand()%1400;
        len[i]=L; sumlen+=L; if(L<mn)mn=L; if(L>mx)mx=L;
    }
    printf("预算 %d 槽 | 最大长度 %d | 页大小 %d | 请求长度 %ld~%ld 平均 %ld\n\n",
           budget, maxseq, page, mn, mx, sumlen/N);

    // ① 传统：每请求预留 maxseq 连续块
    int sT=0; long actT=0;
    for(int i=0;i<N;i++){ if((long)(sT+1)*maxseq<=budget){ sT++; actT+=len[i]; } else break; }
    long resT=(long)sT*maxseq;
    printf("【传统·预留连续块】\n");
    printf("  服务请求数: %d   (每个预留 %d，%d×%d=%ld 占满)\n", sT, maxseq, sT, maxseq, resT);
    printf("  实际用 %ld / 预留 %ld → 利用率 %.1f%%   (浪费 %.1f%%)\n\n",
           actT, resT, 100.0*actT/resT, 100.0*(resT-actT)/resT);

    // ② 分页：每请求按 ceil(len/page) 个页分配
    int sP=0; long allocP=0, actP=0;
    for(int i=0;i<N;i++){ long cost=((len[i]+page-1)/page)*page;
        if(allocP+cost<=budget){ sP++; allocP+=cost; actP+=len[i]; } else break; }
    printf("【PagedAttention·按页分配】\n");
    printf("  服务请求数: %d   (按需分配 %ld 槽，无需预留 maxseq)\n", sP, allocP);
    printf("  实际用 %ld / 分配 %ld → 利用率 %.1f%%   (仅末页零头浪费)\n\n",
           actP, allocP, 100.0*actP/allocP);

    printf(">>> 同样 %d 槽显存：分页服务 %d 个请求 vs 传统 %d 个 = %.1f× 并发\n",
           budget, sP, sT, (double)sP/sT);
    printf(">>> 并发请求数 = 有效 batch → 直接接 topic 04 实验①：batch↑ → 吞吐↑\n");
    return 0;
}
