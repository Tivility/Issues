//use class to write each algorithm just 'cuz notepad++
//class and function could be easily found.
/**************************************************
*******************简单算法与STL*******************
    STL
    输入输出外挂
    大数处理
    大数加法，减法，乘法, 比较, JAtoA

*************************DP************************

************************搜索***********************

**********************数据结构*********************
    并查集 Disjoint_Set
    线段树 Segment_Tree

***********************字符串**********************

************************图论***********************
    链式前向星 Link_Pre_Star
    拓扑序 Topological_Order
    最小生成树 MST(Minimum Spanning Tree)
        Prim Prim
        Kruskal Kruskal
    单源最短距离
        Dij dijkstra
        堆优化Dij priority_dijkstra
        SPFA SPFA
        Floyd + 最小环 Floyd

************************数论***********************
    快速幂运算 Fast_Pow
    欧拉素数筛 Celect_Prime
	Miller-Rabin 素数测试算法
**********************组合数学*********************
    排列组合 Combination
    Lucas定理 Lucas
    矩阵快速幂 Matrix_Fast_Pow
**********************计算几何*********************

***********************博弈论**********************

***************************************************/

/*******************简单算法与STL*******************/
//STL
class STL {
    #pragma comment(linker, "/STACK:1024000000,1024000000")
    #define _CRT_SECURE_NO_WARNINGS
    #pragma _attribute_((optimize("-O2")))
    #ifdef Local
        cout << "time: " << (long long)clock() * 1000 / CLOCKS_PER_SEC << " ms" << endl;
    #endif
    ///C
    #include <stdio.h>
    int __builtin_popcount(unsigned int);//二进制位中1的个数

    #include <stdlib.h>
    int *p; p=(int*)malloc(sizeof(int)); free(p);
    int cmp(const void *a, const void *b ) {
        return *(int*)b - *(int*)a;
    }//降序
    int cmp2(const void *a, const void *b) {
        return *(double*)a>(double*)*b ? 1 : -1;
    }//double

    #include <math.h>
    double fabs(double x);
    double ceil(double x);//向上取整
    double floor(double x);//向下取整
    double round(double x);//最接近x的整数
    //角度制 = 弧度制/180*PI;
    double asin(double arg);//求反正弦 arg∈[-1,1],返回值∈[-pi/2,+pi/2]
    double sin(double arg);//求正弦 arg为弧度,返回值∈[-1, 1]
    double exp(double arg);//求e的arg次方
    double log(double num);//求num的对数,基数为e
    k = (int)(log((double)n) / log(2.0));//2^k==n
    double sqrt(double num);
    double pow(double base,double exp);//求base的exp次方
    memset( the_array, 0, sizeof(the_array) );
    memcpy( the_array, src, sizeof(src));

    #include <string.h>
    int strcmp(const char *str1, const char *str2 );//str1<str2,return负数
    void strcpy(int a[],int b[]);//字符串复制b赋给a
    int strlen(int a[]);
    char *strstr(char *str1, char *str2);//str1中str2串第一次出现的指针,找不到返回NULL

    #include <time.h>
    clock_t clockBegin, clockEnd;
    clockBegin = clock();
    //do something
    clockEnd = clock();
    printf("time:%d\n", clockEnd - clockBegin);

    ///C++
    ///iostream
    #include<iostream>
    cin.getline(s,N,'\n');

    ///sstream
    #include<sstream>
    string s;
    stringstream ss;
    stringstream ss(s);
    ss.str(s);
    ss.clear();
    ss>>a>>b>>c;

    ///string
    #include<string>
    string s,st[maxn],ss(sss);//char sss[N];
    s.c_str();//返回字符数组；
    string st = s.substr(now,len);//截取从now处开始长度为len的子字符串；
    st.assign(s,now,len);//功能同上
    getline(cin,s,'\n');

    ///pair
    pair<int,int> tmp;
    tmp=make_pair<1,2>;
    tmp.first; tmp.second;

    ///vector;
    #include<vector>
    vector<int>v[n];
    vector<int>::iterator it;
    v.front();  v.back();   v.push_back(x); v.assign(begin,end);  v.at(pos);
    v.size();   v.empty();  v.capacity();
    v.resize(num);  v.reserve();
    v.begin();  v.end();
    v.insert(it,x); v.insert(it,n,x);   v.insert(it,first,last);
    v.pop_back();   v.erase(it);    v.erase(first,last);    v.clear();
    v.swap(v);
    //去重，unique删除begin到end之间的相邻重复元素后返回一个新的结尾的迭代器
    v.erase(unique(s.begin(),s.end()),s.end())；

    ///map,key—value关联式容器，可利用key值快速查找记录
    #include<map>
    map<int,string> mp;//int做key值字符串为value
    //string name; mp[name]+=123;
    multimap<int, string> mp;//允许重复
    map<string, int>::iterator it;//用指针处理
    m.insert();
    m.begin();  m.end();    m.find();
    m.clear();  m.erase();
    m.count();  m.empty();  m.size();
    m.swap();
    //返回一个非递减序列[first, last)中的第一个大于等于值val的位置的迭代期。
    lower_bound(first,last,val);
    //返回一个非递减序列[first, last)中第一个大于val的位置的迭代期。
    upper_bound(first,last,val);
    m.max_size();//返回可以容纳的最大元素个数

    ///stack：
    #include<stack>
    stack<int>s;
    s.push(x); s.pop(); s.top(); s.empty();  e.size();

    ///queue;
    #include<queue>
    queue<int>q;
    q.push(x); q.pop(); q.front();  q.back(); q.empty();  q.size();
    //优先队列，优先输出优先级高的队列项:
    struct node {
        int x, y;
        friend bool operator < (node a, node b) {
    //        if(a.y==b.y)return a.x > b.x;//二维判断，按x降序
            return a.y < b.y;//按y降序
        }
    };
    priority_queue <node> q;//定义结构体
    struct mycmp {
        bool operator()(const int &a,const int &b) {
            return a>b;//升序
        }
    };
    priority_queue<int,vector<int>,mycmp> q;
    priority_queue<int,vector<int>,greater<int> > q;//数组升序优先队列
    priority_queue<int,vector<int>,less<int> > q;//数组降序优先队列
    q.top();//返回优先队列对顶元素

    ///set,红黑树的平衡二叉检索树建立，用于快速检索
    #include<set>
    struct cmp {
        bool operator()( const int &a, const int &b ) const{
            return a>b;
        }//降序
    };
    set <int,cmp> st;
    set <int> s;//默认升序
    bool operator<(const node &a, const node &b) {
        return a.y > b.y;
    }//降序
    struct node{
        int x, y;
        friend bool operator < (node a, node b) {
    //        if(a.y==b.y)return a.x > b.x;//二维判断，按x降序
            return a.y > b.y;//按y降序
        }
    } t;
    set<node> st;
    multiset<int> st;//与set不同，允许相同元素
    multiset<int>::iterator it;//取指针
    it=st.begin();
    //如果是:st.erase(*it);删除元素的话，所有的相同元素都会被删除
    st.erase(it);   st.clear();
    st.begin(); st.end(); st.rbegin();
    st.insert();
    st.count(); st.empty(); st.size();  st.find();
    st.equal_range();//返回集合中与给定值相等的上下限的两个迭代器;
    st.swap();
    st.lower_bound();//返回指向(升序时)大于（或等于）某值的第一个元素的迭代器
    //结构体：it = st.lower_bound(tn); printf("%d\n",(*it).x);
    st.upper_bound();//返回大于某个值元素的迭代器

    ///bitset
    #include<bitset>
    unsigned long u;    string str;
    bitset<N> bit;
    bitset<N> bit(u);
    bitset<N> bit(str);
    bitset<N> bit(str,pos);
    bitset<N> bit(str,pos,num);
    bit.set();  bit.set(pos);   bit.reset();    bit.reset(pos);
    bit.flip(); bit.flip(pos); //按位取反
    bit.size(); bit.count();//1的个数
    bit.any();  bit.none(); //有1，无1；return bool
    bit.test(pos) == bit[pos]; //pos处为1？
    u = bit.to_ulong();
    str = bit.to_string(); //str[n-i-i]==bit[i]

    ///algorithm;
    #include<algorithm>
    sort(begin,end);//升序排列；
    bool cmp(int a,int b) {
        return a>b;
    }//降序
    bool cmp(node a,node b) {
    //    if(a.y == b.y)return a.x > b.x;//降序
        return a.y > b.y;//降序
    }
    struct node {
        int x,y;
        friend bool operator < (node a, node b) {
    //        if(a.y==b.y)return a.x < b.x;//二维判断，降序
            return a.y > b.y;//降序
        }
    };
    sort(begin,end,cmp);
    sort(begin,end);//结构体中友元定义排序规则
    stable_sort(begin,end,cmp);//稳定排序
    //去除相邻的重复值(把重复的放到最后)，然后返回去重后的最后一个元素的地址；
    unique(begin,end);
    //在[begin,end)中查找val，返回第一个符合条件的元素的迭代器，否则返回end指针
    find(begin,end,val);
    //count，将返回从start到end 范围之内的序列中某个元素的数量。
    n = count(begin,end,val);
    next_permutation(begin,end);//返回该段序列的字典序下一种排列
}
//输入输出外挂
//大数处理
//大数加法, 减法, 乘法, 比较, JAtoA
/*************************DP************************/
/****背包问题****/
//01背包
//完全背包
//多重背包
//多重背包(单调队列优化)
//二维费用背包
//斜率优化DP
//最大连续子序列之和
//最大子矩阵和
//最长递增子序列(LIS)
//最长公共子序列(LCS)
//数位DP
//树DP
//区间DP
/****Sparse Table****/
//一维RMQ
//二维RMQ
/************************搜索***********************/
//DFS
//BFS(队列解法)
//A*启发式搜索算法
//IDA*迭代深化A*搜索
//Dancing Link
/**********************数据结构*********************/
//并查集
class Disjoint_Set {
    int pre[MAXN];
    int find(int now) {
        return pre[now] == now ? now : (pre[now] = find(pre[now]));
    }
    void Union(int a, int b) {
        pre[find(a)] = find(b);
        return ;
    }
    void init(int n) {
        for (int i = 0; i <= n; ++i)
            pre[i] = i;
        return ;
    }
}
//线段树
class Segment_Tree {
    const unsigned int MAX = 1e5+10;
    unsigned int n, q, ITOR, sl, sr;
    struct RMQ{
        unsigned int l, r, nxl, nxr;
        long long lazy, sum;
    } tree[(MAX<<1)];
    long long Built(RMQ* point) {
        point->lazy = 0;
        if ((point->r - point->l) == 1) {
            point->nxl = 0, point->nxr = 0;
            scanf("%lld", &(point->sum));
            return point->sum;
        }
        unsigned int n = 1;
        while (n < (point->r-point->l)) n <<= 1;
        point->nxl = ++ITOR, point->nxr = ++ITOR;
        tree[point->nxl].l = point->l, tree[point->nxl].r = point->l + (n>>1);
        tree[point->nxr].l = point->l + (n>>1), tree[point->nxr].r = point->r;
        point->sum = (Built(&tree[point->nxl]) + Built(&tree[point->nxr]));
        return point->sum;
    }
    void init(){
        ITOR = 1;
        memset(&tree[0], 0, sizeof(tree[0]));
        tree[1].l = 0, tree[1].r = n, tree[1].lazy = 0;
        Built(&tree[1]);
    }
    long long search(RMQ* now, unsigned int l, unsigned int r) {
        if(now->l == now->r)
            return 0;
        if (l == now->l && r == now->r)
            return (now->sum + now->lazy * (r-l));
        now->sum += (now->lazy * (now->r - now->l));
        tree[now->nxl].lazy += now->lazy, tree[now->nxr].lazy += now->lazy;
        now->lazy = 0;
        if (l >= tree[now->nxl].r)
            return search(&tree[now->nxr], l, r);
        if (r <= tree[now->nxr].l)
            return search(&tree[now->nxl], l, r);
        return (search(&tree[now->nxl], l, tree[now->nxl].r) +
                search(&tree[now->nxr], tree[now->nxr].l, r));
    }
    long long add(RMQ* now, unsigned int l, unsigned int r, long long lazy) {
        if (now->l == now->r) return 0;
        if (l == now->l && r == now->r) {
            now->lazy += lazy;
            return (now->sum + now->lazy * (r-l));
        }
        now->sum += now->lazy * (now->r - now->l);
        now->sum += lazy * (r-l);
        tree[now->nxl].lazy += now->lazy, tree[now->nxr].lazy += now->lazy;
        now->lazy = 0;
        if (r <= tree[now->nxr].l)
            return add(&tree[now->nxl], l, r, lazy);
        if (l >= tree[now->nxr].l)
            return add(&tree[now->nxr], l, r, lazy);
        return (add(&tree[now->nxl], l, tree[now->nxl].r, lazy) +
                add(&tree[now->nxr], tree[now->nxr].l, r, lazy));

    }
    void judge(){
        init();
        scanf("%u%u", &sl, &sr);
        printf("%lld\n", search(&tree[1], sl-1, sr));
        scanf("%u%u%lld", &sl, &sr, &sn);
        add(&tree[1], sl-1, sr, sn);
    }
}
//划分树
//树状数组
//树链剖分
//字典树
//Treap
//伸展树(splay tree)
/****主席树****/
//静态区间k大
//动态区间k大
//区间不相同数数量
//树上路径点权第k大
//哈希表
//ELFhash（字符串哈希函数）
//莫队算法
/***********************字符串**********************/
//字符串最小表示法
//Manacher最长回文子串
//KMP
//扩展kmp
//AC自动机
/****后缀数组****/
//DA倍增算法
//DC3算法
//后缀自动机
/************************图论***********************/
//链式前向星
class Link_Pre_Star {
    struct Edge {
        int to, nxt, w;
    } edge[MAXE << 1]; //双向边
    int head[MAXN], ecnt;
    void init() {
        ecnt = 0;
        memset (head, -1, sizeof (head));
    }
    void add(int pre, int to, int w) { //单向边
        edge[ecnt].nxt = head[pre];
        edge[ecnt].w = w, edge[ecnt].to = to;
        head[pre] = ecnt++;
        //++in_degree[to], ++out_degree[pre];//出度入度
    }
    void _add(int u, int v, int w) {   //双向边
        edge[ecnt].nxt = head[u], edge[ecnt^1].nxt = head[v];
        edge[ecnt].w = edge[ecnt^1].w = w;
        edge[ecnt].to = v, edge[ecnt^1].to = u;
        head[u] = ecnt, head[v] = ecnt^1;
        ecnt += 2;
    }
}
//拓扑序
class Topological_Order { // from 0 to (n-1)
    void init() {
        qit = 0;
        for (int i = 0; i < n; ++i)
            if (ind[i] == 0)
                que[qit++] = i;
    }
    void topological_travel() {
        for (int i = 0; i < qit; ++i)
            for (int j = head[que[i]]; j != -1; j = edge[j].nxt) {
                pre = que[i], to = edge[j].to;
                //node[to] is next node in order
                if (!--ind[to])
                    que[qit++] = to;
            }
    }
}
/****最小生成树****/
//prim
class Prim { //mp[n][n], from 1 to n
    int n, m, u, v, w;
    int mp[MAX_N][MAX_N];
    bool used[MAX_N];
    void getin() {
        memset (mp, 0x3f, sizeof (mp));
        for (int i = 0; i <= n; ++i)
            mp[i][i] = 0;
        for (int i = 0; i < m; ++i) {
            scanf("%d%d%d", &u, &v, &w);
            mp[u][v] = mp[v][u] = min(w, mp[u][v]);
        }
    }
    int prime() {
        int ret = 0;
        memset (used, 0, sizeof (used));
        memset (dis, 0x3f, sizeof (dis));
        dis[1] = 0;
        for (int i = 1; i < n; ++i) {
            int now = 0;
            for (int j = 1; j <= n; ++j)
                if (!used[j] && (now == 0 || dis[j] < dis[now])) now = j;
            used[now] = 1;
            for (int j = 1; j <= n; ++j)
                if (!used[j]) dis[j] = min(dis[j], mp[now][j]);
        }
        for (int i = 2; i <= n; ++i) ret += dis[i];
        return ret;
    }
}
//kruskal
class Kruskal {
    int n, m, u, v, w, ans;
    int pre[MAX];
    struct Edge {
        int u, v, w;
    } edge[MAX * MAX];
    bool cmp(const Edge a, const Edge b) {
        return a.w < b.w;
    }
    void getin() {
        for (int i = 0; i < m; ++i) {
            scanf("%d%d%d", &u, &v, &w);
            edge[m].u = u, edge[m].v = v, edge[m].w = w;
        }
        sort (edge, edge+m, cmp);
    }
    int kurskal(Edge *edge, int m) {
        int now, ret = 0;
        for (int i = 0; i < m; ++i) {
            u = edge[i].u, v = edge[i].v;
            if (find(u) != find(v)) {
                //vtor.push_back(now);
                ret += edge[i].w;
                Union(u, v);
            }
        }
        return ret;
    }
}
//次小生成树
/****单源最短距离****/
//Dijkstra
class Dijkstra { // from 1 to n
    const int MAXN = 1010;
    const double INF = 1e9+7;
    double road[MAXN][MAXN], dis[MAXN];
    bool dis[MAXN];
    int n;
    double dij(int s, int e) {
        for (int i = 1; i <= n; ++i) {
            dis[i] = INF, dis[i] = 0;
        }
        dis[s] = 0.0;
        double mmin; int now;
        while (1) {
            mmin = INF; now = 0;
            for (int i = 1; i <= n; ++i)
                if (!dis[i] && dis[i] < mmin)
                    mmin = dis[i], now = i;
            if (mmin == INF) break;
            dis[now] = 1;
            for (int i = 1; i <= n; ++i) {
                if (dis[now] + road[now][i] < dis[i]){
                    dis[i] = dis[now] + road[now][i];
                }
            }
        }
        return dis[e];
    }
}
//堆优化Dijkstra
class Priority_Dijkstra { //from 1 to n
    struct Node {
        int id;
        bool operator < (const Node &b) const{
            return dis[id] > dis[b.id];
        }
    }node;
    const int INF = Max_Length;
    int dis[MAXN], n, m, pre, to, w;
    bool vis[MAXN];
    priority_queue <Node> que;
    int pri_dij(int st, int ed) {
        while (que.size()) que.pop();
        for (int i = 0; i <= n; ++i) {
            dis[i] = INF, vis[i] = 0;
        }
        int now = st;
        dis[now]= 0, node.id = now, que.push(node);
        while (que.size()) {
            now = que.top().id;
            que.pop();
            if (tois[now]) continue;
            for (int i = head[now]; i != -1; i = edge[i].nxt) {
                to = edge[i].to;
                if (dis[to] > dis[now] + edge[i].w) {
                    dis[to] = dis[now] + edge[i].w;
                    if (!tois[to]) {
                        node.id = to;
                        que.push(node);
                    }
                }
            }
            vis[now] = 1;
        }
        return dis[ed];
    }
}
//Bellman-Ford
//SPFA
class SPFA { //judge Negative Ring
    int dis[MAX_N], cnt[MAX_N];
    bool used[MAX_N], Negative_Ring;
    queue <int> que;
    int spfa (int st, int ed) {
        for (int i = 0; i <= n; ++i)
            used[i] = 0, dis[i] = INF, ;
        while (que.size()) que.pop();
        int now;
        used[st] = 1, dis[st] = 0, cnt[st] = 0;
        Negative_Ring = false;
        que.push(st);
        while (que.size()) {
            now = que.front(), que.pop();
            used[now] = 0;
            for (int i = head[now]; i != -1; i = edge[i].nxt) {
                if (dis[now] + edge[i].w < dis[edge[i].to]) {
                    dis[edge[i].to] = dis[now] + edge[i].w;
                    cnt[edge[i].to] = cht[now] + 1;//Negative Ring
                    if (cnt[edge[i].to] >= n) {
                        Negative_Ring = true;
                        return 0;
                    }
                    if (!used[edge[i].to]) {
                        used[edge[i].to] = 1;
                        que.push(edge[i].to);
                    }
                }
            }
        }
        return dis[ed];
    }
}
//Floyd 全图最短路 + 最小环
class Floyd {
    int d[MAX_N][MAX_N];
    void Floyd() {
        for (int k = 1; k <= n; ++k) {
			for (int i = 1; i < n; ++i)
				for (int j = 1; j < n; ++j)
					minc = min(minc, d[i][j] + map[i][k] + map[k][j]);
            for (int i = 1; i <= n; ++i) 
                for (int j = 1; j <= n; ++j) 
                    d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
        }
    }
}
//fleury 欧拉回路
/****网络流****/
//最大流
//EdmondsKarp
//dinic
//SAP
//ISAP
//最小费用最大流
/****二分图****/
//匈牙利算法
//Hopcroft-Karp算法
//KM算法
//tarjan 强连通分量
/****双连通分量****/
//边双连通分量
//点双连通分量
/****最近公共祖先(LCA)****/
class LCA {
    struct Edge {
    	int to, nxt, w;
    }edge[MAX << 1];
    bool used[MAX];
    int T, n, m, tot, t, u, v, w, to;
    int head[MAX], ind[MAX], f[MAX][20], deep[MAX];
    long long dis[MAX];
    queue <int> que;
    void addedge(int from, int to, int w = 0) {
    	edge[tot].to = to, edge[tot].w = w;
    	edge[tot].nxt = head[from], head[from] = tot++;
    }
    void init() {
    	tot=  0;
    	memset (head, -1, sizeof (head));
    	memset (dis, 0, sizeof (dis));
    	memset (deep, 0, sizeof (deep));
    	memset (used, 0, sizeof (used));
    //	memset (ind, 0, sizeof (ind));
    }
    void bfs(int h) {
    	int now;
    	while (que.size()) que.pop();
    	que.push(h), used[h] = 1;
    	deep[h] = 0, dis[h] = 0;
    	while (que.size()) {
    		now = que.front(), que.pop();
    		used[now] = 1;
    		for (int i = head[now]; i != -1; i = edge[i].nxt) {
    			to = edge[i].to;
    			if (used[to]) continue;
    			dis[to] = dis[now] + edge[i].w;
    			deep[to] = deep[now] + 1;
    			f[to][0] = now;
    			for (int j = 1; j <= t; ++j)
    				f[to][j] = f[f[to][j-1]][j-1];
    			que.push(to), used[to] = 1;
    		}
    	}
    	return ;
    }
    int lca(int x, int  y) {
    	if (deep[x] > deep[y]) swap(x, y);
    	for (int i = t; i >= 0; --i)
    		if (deep[f[y][i]] >= deep[x])
    			y = f[y][i];
    	if (x == y) return x;
    	for (int i = t; i >= 0; --i)
    		if (f[x][i] != f[y][i])
    			x = f[x][i], y = f[y][i];
    	return f[x][0];
    }
    void getin() {
    	cin >> n >> m;//n points, m queries
    	init();
    	t = (int) (log(n) / log(2)) + 1;

    	for (int i = 0; i < n-1; ++i) {
    		cin >> u >> v >> w;
    		addedge(u, v , w), addedge(v, u, w);
    		//++ind[v]; //Directed graph
    	}
    	/** Directed graph
    	for (int i = 1; i <= n; ++i) {
    		if (ind[i] == 0) {
    			bfs(i);
    			break;
    		}
    	}
    	**/
    	bfs(1);
    	return ;
    }
}
//tarjan
//ST-RMQ在线算法
/************************数论***********************/
//Fibonacci Number
//Greatest Common Ditoisor 最大公约数,欧几里德算法
//Lowest Common Multiple 最小公倍数
//扩展欧几里德算法
//快速幂运算
class Fast_Pow {
    long long fastpow(long long a, long long b, int mod) {
        long long res=1;
        a %= mod;
        for ( ; b; b >>= 1) {
            if (b & 1)
                (res *= a) %= mod;
            (a *= a) %= mod;
        }
        return res;
    }
}
/****质数判断****/
//欧拉素数筛
class Celect_Prime {
    bool isprime[MAXN]={0};
    int prime[MAXP]={0}, p=0;
    void get_prime (int n) {
        memset (isprime, true, sizeof ( isprime ) );
        isprime[0] = isprime[1] = false;
        for (int k = 2; k <= n; k++ ) {
            if (isprime[k]) prime[p++] = k;
            for (int i = 0; i < p; i++) {
                if (k * prime[i] > n)
                    break;
                isprime[k * prime[i]] = false;
                if (k % prime[i] == 0)
                    break;
            }
        }
    }
}
//简单Prime判断
//Sietoe Prime  素数筛选法
//Miller-Rabin 素数测试算法
class Miller_Rabin {
	//from Air 寒域
    //  Miller_Rabin判断素数
    bool Miller_Rabin(long long int n) {
        return Miller_Rabin(n, 40);
    }
    //  Miller_Rabin判断素数
    bool Miller_Rabin(long long int n, long long int S) {
        if (n == 2)return true;
        if (n < 2 || !(n & 1))return false;
        int t = 0;
        long long int a, x, y, u = n - 1;
        while ((u & 1) == 0) t++, u >>= 1;
        for (int i = 0; i < S; i++) {
            a = rand() % (n - 1) + 1;
            x = modular_exp(a, u, n);
            for (int j = 0; j < t; j++) {
                y = modular_multi(x, x, n);
                if (y == 1 && x != 1 && x != n - 1)
                    return false;
                x = y;
            }
            if (x != 1)
                return false;
        }
        return true;
    }
}
//唯一分解定理，因子分解求和
//pollard_rho 算法  质因数分解
//欧拉函数
//约瑟夫环
//高斯消元  开关问题
/**********************组合数学*********************/
//排列组合
class Combination {

}
//Lucas定理
class Lucas {//C(n, m) % mod;
    const int MAX = 2e5+10;
    long long jc[MAX], jcinv[MAX];
    void init() {
        jc[0] = 1, jcinv[0] = 1;
        for (int i = 1; i < MAX; ++i) {
            jc[i] = jc[i-1] * i % mod;
            jcinv[i] = fastpow(jc[i], mod - 2);
        }
    }
    int C(int n, int m) {
        long long ret = 1, a, b;
        while (n && m) {
            a = n % p, b = m % p;
            if (a < b) return 0; //分子含模, 一定得0
            (ret *= (jc[a] * jcinv[b] % mod * jcinv[a-b] % mod)) % mod;
            n /= p, m /= p;
        }
        return ret;
    }
}
//全排列
//错排公式
//母函数
//整数划分
//康托展开
//逆康托展开
//Catalan Number
//Stirling Number(Second Kind)
//容斥原理
//矩阵快速幂
class Matrix_Fast_Pow {
    const int MAX_M = 10;
    const int mod = 1e9+7;
    struct Martix {
        int ary[MAX_M][MAX_M];
        int row, column;
        void clear() {
            memset(ary, 0, sizeof(ary));
            row = 0, column = 0;
        }
        void getin(int row, int column) {
            for (int i = 0; i < row; ++i)
                for (int j = 0; j < column; ++j)
                    cin >> ary[i][j];
        }
        void output() {
            for (int i = 0; i < row; ++i) {
                for (int j = 0; j < column; ++j)
                    cout << ary[i][j] << " ";
                cout << endl;
            }
        }
        Martix operator + (const Martix &b) const {
            Martix tmp;
            tmp.row = row, tmp.column = column;
            for (int i = 0; i < row; ++i)
                for (int j = 0; j < column; ++j)
                    tmp.ary[i][j] = ary[i][j] + b.ary[i][j];
        }
        Martix operator - (const Martix &b) const {
            Martix tmp;
            tmp.row = row, tmp.column = column;
            for (int i = 0; i < row; ++i)
                for (int j = 0; j < column; ++j)
                    tmp.ary[i][j] = ary[i][j] - b.ary[i][j];
        }
        Martix operator * (const Martix &b) const {
            Martix tmp;
            tmp.clear();
            if (column != b.row)
                return tmp;
            tmp.row = row, tmp.column = b.column;
            for (int i = 0; i < row; ++i)
                for (int j = 0; j < b.column; ++j)
                    for (int k = 0; k < column; ++k) {
                        long long t = (((long long)(ary[i][k]%mod) * (b.ary[k][j]%mod)) % mod);
                        tmp.ary[i][j] += (int)t;
                        while (tmp.ary[i][j] >= mod) tmp.ary[i][j] -= mod;
                    }
            return tmp;
        }
    } matrix, mt;
    Martix matpow(Martix tmp, int n, int mod) {
        Martix ret;
        ret.row = tmp.row, ret.column = tmp.column;
        for (int i = 0; i < tmp.row; ++i)
            for(int j = 0; j < tmp.row; ++j)
                ret.ary[i][j] = (i == j);
        while (n) {
            if (n & 1)
                ret = (ret * tmp);
            tmp = (tmp * tmp), n >>= 1;
        }
        return ret;
    }
}
/**********************计算几何*********************/
//坐标向量
/****三角形****/
//三角形面积公式
//三角形内切圆半径公式
//三角形外接圆半径公式
//圆内接四边形面积公式
//多边形面积
//凸包
//三分求极值
/***********************博弈论**********************/
//巴什博弈：
//威佐夫博弈：
//nim博弈：
//k倍减法博弈：
//sg函数



/*WISH GOD BLESS US*/
