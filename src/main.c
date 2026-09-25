#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

typedef uint8_t  byte;
typedef float  f32;
typedef double f64;
typedef int32_t i32;


// cc src/main.c -o mlp -O3 -march=native -funroll-loops

#define ARENA_ALIGN ((size_t)16)

typedef struct MemoryArena {
    byte* Base;
    size_t Capacity;
    size_t Used;
} MemoryArena;

MemoryArena* ArenaCreate(size_t capacity) {
    MemoryArena* a = calloc(1, capacity + sizeof(MemoryArena));
    if (a == NULL) {fprintf(stderr, "Failed to allocate %zu bytes for arena\n", capacity); abort(); }
    a->Base = (byte *)(a + 1); // data starts right after header
    a->Capacity = capacity;
    a->Used = 0;
    return a;
}

void ArenaReset(MemoryArena* a) {
    a->Used = 0;
}

void ArenaDestroy(MemoryArena* a) {
    free(a);
}

size_t ArenaGetAlignOffset(MemoryArena* a, size_t alignment) {
    assert(alignment > 0 && (alignment & (alignment - 1)) == 0 && "alignment must be a power of 2");
    size_t current_address = (size_t)(a->Base + a->Used);
    size_t mask = alignment - 1;
    if (current_address & mask) {
        return alignment - (current_address & mask);
    }
    return 0;
}

void* ArenaPushAligned(MemoryArena* a, size_t size, size_t alignment) {
    size_t offset = ArenaGetAlignOffset(a, alignment);
    size_t total = size + offset;
    if (total > a->Capacity - a->Used) {
        fprintf(stderr, "Arena out of capacity! requested %zu capacity %zu\n", total, a->Capacity);
        abort();
    }
    void* result = a->Base + a->Used + offset;
    a->Used += total;
    return result;
}

void* ArenaPush(MemoryArena* a, size_t size) {
    return ArenaPushAligned(a, size, ARENA_ALIGN);
}

void ArenaLog(MemoryArena *a) {
    printf("arena used: %zu cap: %zu (%.3f%%) current ptr: %p\n", 
        a->Used, a->Capacity, 
        (f64)a->Used / (f64)a->Capacity * 100.0,
        a->Base + a->Used);
}


typedef struct Matrix {
    int Rows, Cols;
    f64* Data; // flat array, row major
} Matrix;

typedef struct Vector {
    int Length;
    f64* Data;
} Vector;

typedef struct MLP {
    Matrix* E;
    Matrix* W1;
    Vector* b1;
    Matrix* W2;
    Vector* b2;
    int ContextLength;
} MLP;


typedef struct GradState {
    Vector* X_embed;  // output of E, input to W1
    Vector* X_hidden; // output of W1, input to activation
    Vector* A;        // output of activation, input to W2
    Vector* Probs;    // output of softmax
    
    Matrix* Grad_E;
    Matrix* Grad_W1;
    Vector* Grad_b1;
    Matrix* Grad_W2;
    Vector* Grad_b2;
} GradState;

f64 RndF64() {
    return rand() / (f64)RAND_MAX;
}


typedef struct ActivationStats {
    Vector* Sum; // sum of A[i] over all foward passes
    Vector* ZeroCount; // count of A[i] == 0
    int ActivationsTotal; // forward pass count
} ActivationStats;

void PrintActivationStats(ActivationStats* s) {
    if (s->ActivationsTotal == 0) return;
    int n = s->Sum->Length;
    printf("--- activation stats over %d forwards ---\n", s->ActivationsTotal);
    int dead = 0;
    for (int i = 0; i < n; i++) {
        f64 mean      = s->Sum->Data[i] / s->ActivationsTotal;
        f64 zero_frac = s->ZeroCount->Data[i] / s->ActivationsTotal;
        if (zero_frac > 0.99) dead++;
        printf("neuron %2d: mean=%8.4f zero_frac=%.3f%s\n",
               i, mean, zero_frac, zero_frac > 0.99 ? "  <-- DEAD" : "");
    }
    printf("dead neurons: %d / %d\n", dead, n);
}


Matrix* MatrixCreate(MemoryArena* a, int rows, int cols, f64 init_scale) {
    Matrix* m = ArenaPush(a, sizeof(Matrix));
    m->Data = ArenaPush(a, sizeof(f64) * rows * cols);
    m->Rows = rows;
    m->Cols = cols;

    for (int i = 0; i < rows*cols; i++) {
        m->Data[i] = init_scale != 0.0 
        ? (RndF64() * 2 -1) * init_scale 
        : init_scale;
    }

    return m;
}

Vector* VectorCreate(MemoryArena* a, int length) {
    Vector* v = ArenaPush(a, sizeof(Vector));
    v->Data = ArenaPush(a, sizeof(f64) * length);
    v->Length = length;
    memset(v->Data, 0, sizeof(f64) * length);
    return v;
}

MLP* MLP_Create(MemoryArena* a, int vocab_size, int context_length, int n_embed, int hidden_dim) {
    MLP* mlp = ArenaPush(a, sizeof(MLP));
    mlp->E = MatrixCreate(a, vocab_size, n_embed, 0.01);
    mlp->W1 = MatrixCreate(a, hidden_dim, n_embed * context_length, 0.01);
    mlp->b1 = VectorCreate(a, hidden_dim);
    mlp->W2 = MatrixCreate(a, vocab_size, hidden_dim, 0.01);
    mlp->b2 = VectorCreate(a, vocab_size);
    mlp->ContextLength = context_length;
    return mlp;
}

GradState* GradStateCreate(MemoryArena* a, MLP* model) {
    GradState* s = ArenaPush(a, sizeof(GradState));
    s->Grad_E = MatrixCreate(a, model->E->Rows, model->E->Cols, 0.0);
    s->Grad_W1 = MatrixCreate(a, model->W1->Rows, model->W1->Cols, 0.0);
    s->Grad_b1 = VectorCreate(a, model->b1->Length);
    s->Grad_W2 = MatrixCreate(a, model->W2->Rows, model->W2->Cols, 0.0);
    s->Grad_b2 = VectorCreate(a, model->b2->Length);

    s->X_embed = VectorCreate(a, model->E->Cols * model->ContextLength);
    s->X_hidden = VectorCreate(a, model->W1->Rows);
    s->A = VectorCreate(a, s->X_hidden->Length);
    s->Probs = VectorCreate(a, model->W2->Rows);
    return s;
}

f64 MLP_Forward(MLP *m, const int* token_ids, int y, GradState* grad_state, ActivationStats* act_stats) {
    //assert(token_idx >= 0 && token_idx < m->E->Rows && "invalid index");

    // embed
    f64* x_embed = grad_state->X_embed->Data;
    int n_embed = m->E->Cols;
/*     for (int c = 0; c < m->E->Cols; c++)  {
        x_embed[c] = m->E->Data[token_idx * m->E->Cols + c];
    } */
    for (int t = 0; t < m->ContextLength; t++) {
        int tkn = token_ids[t];
        for (int c = 0; c < n_embed; c++) {
            x_embed[t * n_embed + c] = m->E->Data[tkn * n_embed + c];
        }
    }

    // linear 1
    f64* x_hidden = grad_state->X_hidden->Data;
    for (int row = 0; row < m->W1->Rows; row ++) {
        f64 dot = 0.0;
        for (int col = 0; col < m->W1->Cols; col ++) {
            dot += m->W1->Data[row * m->W1->Cols + col] * x_embed[col];
        }
        x_hidden[row] = dot + m->b1->Data[row];
    }

    // activation
    f64* A = grad_state->A->Data;
    for (int i = 0; i < m->W1->Rows; i++) {
        A[i] = x_hidden[i] > 0.0 ? x_hidden[i] : 0.0;  // ReLU
    }

    if (act_stats != NULL) {
        assert(act_stats->Sum->Length == grad_state->A->Length && "stats and activations mismatch");
        for (int i = 0; i < grad_state->A->Length; i++) {
            f64 v = A[i];
            act_stats->Sum->Data[i] += v;
            if (v == 0.0) act_stats->ZeroCount->Data[i]++;
        }
        act_stats->ActivationsTotal++;
    }

    assert(m->W1->Rows == m->W2->Cols && "shape mismatch! W1 rows mustbe same as W2 cols");
    // linear 2 (out)
    f64 logits[m->W2->Rows];
    for (int row = 0; row < m->W2->Rows; row ++) {
        f64 dot = 0.0;
        for (int col = 0; col < m->W2->Cols; col ++) {
            dot += m->W2->Data[row * m->W2->Cols + col] * A[col];
        }
        logits[row] = dot + m->b2->Data[row];
    }


    // softmax
    f64* probs = grad_state->Probs->Data;
    f64 exp_sum = 0.0;
    f64 max_val = logits[0];
    for (int i = 0; i < m->W2->Rows; i ++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    for (int i = 0; i < m->W2->Rows; i ++) {
        probs[i] = exp(logits[i] - max_val);
        exp_sum += probs[i];
    }
    for (int i = 0; i < m->W2->Rows; i ++) {
        probs[i] = probs[i] / exp_sum;
    }
    
    f64 loss = -log(probs[y]);
    return loss;
}

void MLP_Backward(MLP* model, const int* token_ids, int y, GradState* grad_state) {

    // token_idx => E -> x_embed => W1(x_embed) + b1 -> x_hidden => Relu(x_hidden) -> a => W2(a) + b2 -> logits => softmax(logits) -> probs


    // dL wrt to dL, how does a change in loss affect the loss?
    // dL/dL = 1

    // dL wrt to probs, how does a change in the probs affect the loss
    // dL/dprobs = -1/probs[y] * 1 for correct label (y) otherqwise 0.0
    // (!) but skip full chain rule through softmax as it simplifies 
    //     to what is below
    
    // dL wrt to logits, how does a change in the logits affect the loss
    // for correct label (y) we have target prob 1.0, for the others 0.0
    // for correct label the higher the logit the lower the loss, so grad is negative
    // for the right label we want higher logits, and for the other lower logits
    // we want stronger signal the further away from the target prob
    // probs[y] - 1 gives us right direction (negative) AND magnitute for the gardient
    // dL/dlogits = i == y ? probs[i] - 1 : probs[i]
    assert(model->W2->Rows == grad_state->Probs->Length && "w2 rows != probs length, shape mismatch");
    f64 dLdlogits[model->W2->Rows]; // (vocab_size, )
    for (int i = 0; i < model->W2->Rows; i ++) {
        dLdlogits[i] = i == y ? grad_state->Probs->Data[i] - 1 :  grad_state->Probs->Data[i];
    }

    // dL wrt to W2 and b2 paramaters, how does the a change in those parameters affect the loss
    // dL/dW2

    // dy/dW2
    // gradient of each weight elemnt in matirx is just x if we think f(w) = x * w + b
    // grad of each bias element is just 1 as it's addtions
    // so we chain these local grads with the upstream dlDlogits grads to get dL/W
    // dLdW2[ij] = a_i * dLdlogits[i]
    // dL/db2[i] = 1 * dLdlogits[i]
    assert(grad_state->A->Length == model->W2->Cols && "w2 cols != activations length, shape mismatch");
    for (int r = 0; r < model->W2->Rows; r ++) {
        for (int c = 0; c < model->W2->Cols; c++) {
            grad_state->Grad_W2->Data[r * model->W2->Cols + c] = grad_state->A->Data[c] *  dLdlogits[r];
        }
    }
    // b2
    assert(model->b2->Length == model->W2->Rows && "b2 W2 shape mismatch");
    for (int i = 0; i < model->b2->Length; i ++) {
        grad_state->Grad_b2->Data[i] = dLdlogits[i];
    }

    // dL/da, loss wrt input activation, how does a change in input change the loss
    // want it because we need to flow the gradient upward ot the other layers
    // for scalar functions f(a) = w * a + b, local derivaite is w
    // so with chain rule dL/da woudl be: w * dlogits[i]
    // but each elemnt it out logits is a sum of products across the row
    // (logit[i] = sum(w_ij * a_j))
    // so we need the derivative of a sum, which is a sum of derivatives
    // dL/da[j] = sum(w_ij * dlogits[i])
    // where i is row index and j is col index
    // (inputs of linear)
    f64 dLda[model->W2->Cols]; // (hidden, )
    for (int c = 0; c < model->W2->Cols; c++) {
        f64 sum = 0.0;
        for (int r = 0; r < model->W2->Rows; r ++) {
            sum += model->W2->Data[r * model->W2->Cols + c] * dLdlogits[r];
        }
        dLda[c] = sum;
    }


    // relu
    assert(grad_state->X_hidden->Length == model->W2->Cols && "x_hidden lenght != W2 cols, shape mismatch");
    f64 dLdhidden[model->W2->Cols]; // (hidden, )
    assert(grad_state->X_hidden->Length == model->W2->Cols && "grad state x hidden != W2 cols, shape mismatch");
    for (int i = 0; i < model->W2->Cols; i++) {
        dLdhidden[i] = grad_state->X_hidden->Data[i] > 0.0 ? dLda[i] : 0.0;
    }

    // W1
    assert(grad_state->X_embed->Length == model->W1->Cols && "x_embed length !+ w1 cols, shape mismatch");
    for (int r = 0; r < model->W1->Rows; r ++) {
        for (int c = 0; c < model->W1->Cols; c++) {
            grad_state->Grad_W1->Data[r * model->W1->Cols + c] = grad_state->X_embed->Data[c] * dLdhidden[r];
        }
    }
    // b1
    assert(grad_state->Grad_b1->Length == model->b1->Length && "grad b1 length != model b1 length, shape mismatch");
    assert(model->b1->Length == model->W1->Rows && "b2 W2 shape mismatch");
    for (int i = 0; i < model->b1->Length; i ++) {
        grad_state->Grad_b1->Data[i] = dLdhidden[i];
    }

    // dLdembed (input of linear)
    f64 dLdembed[model->W1->Cols]; // (n_embed * ctx_len, )
    for (int c = 0; c < model->W1->Cols; c++) {
        f64 sum = 0.0;
        for (int r = 0; r < model->W1->Rows; r ++) {
            sum += model->W1->Data[r * model->W1->Cols + c] * dLdhidden[r];
        }
        dLdembed[c] = sum;
    }

    // gradients flow from the input to W1 backwards to embedding, so embedgrad <- xembedgrad
    // xembed is of shape (n_embed, ) so only n_embed elemnts in the embedding layer
    // receive the gradients, precisely the ones for the row that is mapped to the token idx
    // assert(token_id < model->E->Rows && token_id >= 0 && "token id out of bounds");
    int n_embed = model->E->Cols;
    for (int t = 0; t < model->ContextLength; t++) {
        int token_id = token_ids[t];
        for (int c = 0; c < n_embed; c++) {
            grad_state->Grad_E->Data[token_id * n_embed + c] += dLdembed[t * n_embed + c]; // aggregate grads
        }
    }
}


// c is a joy
typedef struct Dataset {
    char *Blob;
    i32 *Offsets; // pointer ofsets, Offsets[i] = start of string i in Blob
    i32 TotalLen;
    i32 StringsCount;
    i32 MaxLen;
    i32 MinLen;
    i32 AvgLen;
} Dataset;

Dataset* DatasetLoad(MemoryArena* a, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        int err = errno;
        fprintf(stderr, "fopen failed: %s (errno=%d)\n", strerror(err), err);
        abort();
    }
    
    char line[64];
    int count = 0;
    int total_len = 0;
    int max_len = 0;
    int min_len = INT_MAX;
    
    while (fgets(line, sizeof(line), f)) {
        size_t len = strcspn(line, "\n");
        count++;
        total_len += (int)len;
        if ((int)len > max_len) max_len = (int)len;
        if ((int)len < min_len) min_len = (int)len;
    }
    
    if (count == 0) {
        fclose(f);
        fprintf(stderr, "empty file?\n");
        abort();
    }
    rewind(f);
    
    Dataset* ds = ArenaPush(a, sizeof(Dataset));
    ds->Offsets = ArenaPush(a, sizeof(i32) * count);
    ds->Blob = ArenaPush(a, total_len + count);  // +count for nul terminators
    
    ds->StringsCount = count;
    ds->TotalLen = total_len + count;
    ds->MaxLen = max_len;
    ds->MinLen = min_len;
    ds->AvgLen = total_len / count;
    
    char* current_string = ds->Blob;
    int i = 0;
    while (fgets(line, sizeof line, f)) {
        size_t len = strcspn(line, "\n");
        ds->Offsets[i] = (i32)(current_string - ds->Blob);
        memcpy(current_string, line, len);
        current_string[len] = '\0';
        current_string += len + 1;
        i++;
    }

    printf("loaded dataset %s. total entries: %d, minlen: %d maxlen: %d avg_len: %d total_len: %d\n", 
        path,
        count, min_len, max_len, ds->AvgLen, total_len);
    
    fclose(f);
    return ds;
}

char* DatasetGetStr(Dataset *ds, int idx) {
    if (idx < 0 || idx >= ds->StringsCount) {
        fprintf(stderr, "DatasetGetStr idx out of range %d\n", idx);
        abort();
    }
    return ds->Blob + ds->Offsets[idx];
}

typedef struct Split {
    i32* TrainIndices;
    i32 TrainCount;
    i32* TestIndicies;
    i32 TestCount;
} Split;

static void Shuffle(i32* arr, int length) {
    for (int i = length - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        i32 temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

Split CreateSplit(MemoryArena* a, Dataset* ds, f64 test_fraction) {
    assert(test_fraction >= 0.0 && test_fraction < 1.0 && "invalid value for test fraction");
    int count = ds->StringsCount;

    i32* all = ArenaPush(a, sizeof(i32) * count);
    for (int i = 0; i < count; i++) all[i] = i;
    Shuffle(all, count);

    int test_count = (int)(count * test_fraction);
    if (test_count > 1000) test_count = 1000;
    if (test_count < 1) test_count = 1;
    int train_count = count - test_count;

    Split s;
    s.TrainIndices = all;
    s.TrainCount = train_count;
    s.TestIndicies = all + train_count;
    s.TestCount = test_count;

    return s;
}

char* GetTrainSample(Dataset* ds, Split* split, int i) {
    return DatasetGetStr(ds, split->TrainIndices[i]);
}
char* GetTestSample(Dataset* ds, Split* split, int i) {
    return DatasetGetStr(ds, split->TestIndicies[i]);
}

#define VOCAB_SIZE 28
#define TKN_BLANK (VOCAB_SIZE - 1)
#define USED_TOKENS (VOCAB_SIZE - 1)
#define CTX_LEN 5


byte stoi[256];
char itos[USED_TOKENS];
const byte EOS = '.';

static void BuildVocab(Dataset* ds) {
    memset(stoi, 0, sizeof(stoi));
    // token 0 is '.' (BOS + EOS)
    stoi[(byte)EOS] = 0;
    itos[0] = EOS;
    // 1..26 = 'a'..'z'
    for (int i = 0; i < 26; i++) {
        stoi[(byte)('a' + i)] = (byte)(i + 1);
        itos[i + 1]           = (char)('a' + i);
    }
}

static inline int PaddedAt(const char* sample, int sample_len, int pos) {
    if (pos < 0) return TKN_BLANK;
    if (pos == 0) return stoi[(byte)EOS]; // begin
    if (pos == sample_len + 1) return stoi[(byte)EOS]; // end
    return stoi[(byte)sample[pos - 1]];
}


f64 EvalLoss(MLP* model, GradState* gs, Dataset* ds, int* indices, int count) {
    f64 total = 0.0;
    int tokens = 0;
    int context[CTX_LEN];
    for (int w = 0; w < count; w++) {
        const char* word = DatasetGetStr(ds, indices[w]);
        int len = (int)strlen(word);
        for (int i = 0; i <= len; i++) {
            int y = (i == len) ? stoi[(byte)EOS] : stoi[(byte)word[i]];
            for (int k = 0; k < CTX_LEN; k++) {
                int pos = i - (CTX_LEN - 1 - k);
                context[k] = PaddedAt(word, len, pos);
            }
            total += MLP_Forward(model, context, y, gs, NULL);
            tokens++;
        }
    }
    return total / (f64)tokens;
}

void SampleNames(MLP* model, GradState* gs, int num_samples) {
    int context[CTX_LEN];
    int null_y = 0;

    for (int k = 0; k < num_samples; k++) {
        int sampled_tokens[128];
        int seq_len = 1;

        sampled_tokens[0] = stoi[(byte)EOS]; // start with EOS token

        for (int step = 0; step < 30; step++) {
            for (int t = 0; t < CTX_LEN; t++) {
                int pos = seq_len - 1 - (CTX_LEN - 1 - t);
                context[t] = (pos < 0) ? TKN_BLANK : sampled_tokens[pos];
            }

            MLP_Forward(model, context, null_y, gs, NULL);

            // the softmax includes BLANK
            // we never use BLANK as y, never train on it
            // exclude it from the prob distri by scaling the roll
            f64 total = 0.0;
            for (int j = 0; j < USED_TOKENS; j++) total += gs->Probs->Data[j];

            f64 roll = RndF64() * total; // roll random point on slider [0..total]
            f64 running_total = 0.0; // how filled the slider is
            int next_tkn = 0;
            for (int j = 0; j < USED_TOKENS; j++) {
                running_total += gs->Probs->Data[j]; // fill slider...
                if (roll <= running_total) {  // ...till we capture our random point
                    next_tkn = j; 
                    break; 
                }
            }
            if (next_tkn == stoi[(byte)EOS]) break;
            putchar(itos[next_tkn]);
            sampled_tokens[seq_len++] = next_tkn;
        }
        putchar('\n');
    }
}

int main(int argc, char *argv[]) {
    srand((uint32_t)123);

    MemoryArena* main_arena = ArenaCreate((size_t)(10 << 20));
    ArenaLog(main_arena);

    /* 

    train: steps=320000 lr=0.001 vocab=28 hidden=64 n_embed=16 seq_len=5
    final train loss: 2.0586
    final test  loss: 2.0824
    samples:
        xiangrayalin
        avuma
        wayler
        malann
        isabellu
        jari
        adija
        kadeem
        kennel
        aayssa

    train: steps=320000 lr=0.001 vocab=28 hidden=64 n_embed=16 seq_len=3
    final train loss: 2.1371
    final test  loss: 2.1652
    samples:
        jayi
        limenik
        abrie
        josa
        rus
        kayla
        ara
        elosi
        cen
        neioro

    
    train: steps=320000 lr=0.020 vocab=27 hidden=64 n_embed=16 seq_len=1
    final train loss: 2.4691
    final test  loss: 2.4647
    samples:
        malbronkaman
        caiyzaleli
        kerdieraet
        shisieey
        maeeleker
        kowevkamulafi
        fo
        kabeitt
        ta
        rryra
    */

    f64 lr = 0.02;
    const int vocab_size = VOCAB_SIZE;
    const int hidden_dim = 64;
    const int n_embed    = 16;
    const int seq_len =  CTX_LEN; 
    f64 expected_loss = -log(1/(f64)vocab_size);
    MLP* model = MLP_Create(main_arena, vocab_size, seq_len, n_embed, hidden_dim);
    ArenaLog(main_arena);
    GradState* grad_state = GradStateCreate(main_arena, model);
    ArenaLog(main_arena);

    ActivationStats act_stats;
    act_stats.Sum = VectorCreate(main_arena, model->W1->Rows);
    act_stats.ZeroCount = VectorCreate(main_arena, model->W1->Rows);
    act_stats.ActivationsTotal = 0;

    #define NUM_W 3
    Matrix* m_params[NUM_W] = {model->E, model->W1, model->W2};
    Matrix* m_grads[NUM_W] = {grad_state->Grad_E, grad_state->Grad_W1, grad_state->Grad_W2};

    #define NUM_B 2
    Vector* v_params[NUM_B] = {model->b1, model->b2};
    Vector* v_grads[NUM_B] = {grad_state->Grad_b1, grad_state->Grad_b2};

    Dataset* ds = DatasetLoad(main_arena, "names.txt");
    char* sample = ds->Blob + ds->Offsets[0];
    printf("example string: %s\n", sample);
    ArenaLog(main_arena);

    BuildVocab(ds);

    Split split = CreateSplit(main_arena, ds, 0.1);
    
    const int max_train_steps = 320000;

    int context[CTX_LEN];
    for (int step = 0; step < max_train_steps; step++) {
        //const char* s = DatasetGetStr(ds, step % ds->StringsCount);
        // printf("step %d picked sample: %s len: %d\n", step, sample, sample_len);

        int sample_idx = rand() % split.TrainCount;
        const char* sample = GetTrainSample(ds, &split, sample_idx);
        int sample_len = (int)strlen(sample);

        // '.emma.'
        for (int i = 0; i <= sample_len; i ++) {
            // . -> blank blank .
            // y = e
            // e -> blank . e
            // y = m
            // m -> . e m
            // y = m
            int y = (i == sample_len) ? stoi[(byte)EOS]
                                    : stoi[(byte)sample[i]];

            for (int t = 0; t < CTX_LEN; t++) {
                int pos = i - (CTX_LEN - 1 - t);
                context[t] = PaddedAt(sample, sample_len, pos);
            }

            f64 loss = MLP_Forward(model, context, y, grad_state, &act_stats);

            if (step < 20 || step % 1000 == 0 || step == max_train_steps -1) {
                printf("step: %d loss: %.3f\n", step, loss);
            }
    
            MLP_Backward(model, context, y, grad_state);
    
            // gd
            for (int mi = 0; mi < NUM_W; mi ++){
                Matrix* m = m_params[mi];
                Matrix* g = m_grads[mi];
                for (int i = 0; i < m->Cols * m->Rows; i ++) {
                    m->Data[i] += -lr * g->Data[i];
                    g->Data[i] = 0.0;
                }
            }
            for (int vi = 0; vi < NUM_B; vi ++){
                Vector* v = v_params[vi];
                Vector* g = v_grads[vi];
                for (int i = 0; i < v->Length; i ++) {
                    v->Data[i] += -lr * g->Data[i];
                    g->Data[i] = 0.0;
                }
            }

        }
        lr = step < 100000 ? 0.02 : 0.001;
    }
    
    ArenaLog(main_arena);
    PrintActivationStats(&act_stats);
    
    printf("expected rnd init loss: %.4f\n", expected_loss);
    printf("size of MLP: %zu\n", sizeof(MLP));

    printf("train: steps=%d lr=%.3f vocab=%d hidden=%d n_embed=%d seq_len=%d\n",
       max_train_steps, lr, vocab_size, hidden_dim, n_embed, seq_len);
    f64 train_loss = EvalLoss(model, grad_state, ds, split.TrainIndices, split.TrainCount);
    f64 test_loss  = EvalLoss(model, grad_state, ds, split.TestIndicies, split.TestCount);
    printf("final train loss: %.4f\n", train_loss);
    printf("final test  loss: %.4f\n", test_loss);


    printf("samples:\n");
    SampleNames(model, grad_state, 10);

    #if 0
    f64 logits[4] = {-0.02, -0.01, 0.01, 0.02};
    for (int i = 0; i < 4; i++) {
        f64 exp1 = exp(logits[i]);
        f64 loss1 = -log(exp1);
        printf("logit: %.4f exp %.4f: loss: %.4f\n", logits[i], exp1, loss1);
    }
    #endif

    return 0;
}