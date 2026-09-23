#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

typedef uint8_t  byte;
typedef float  f32;
typedef double f64;


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
    return mlp;
}

GradState* GradStateCreate(MemoryArena* a, MLP* model) {
    GradState* s = ArenaPush(a, sizeof(GradState));
    s->Grad_E = MatrixCreate(a, model->E->Rows, model->E->Cols, 0.0);
    s->Grad_W1 = MatrixCreate(a, model->W1->Rows, model->W1->Cols, 0.0);
    s->Grad_b1 = VectorCreate(a, model->b1->Length);
    s->Grad_W2 = MatrixCreate(a, model->W2->Rows, model->W2->Cols, 0.0);
    s->Grad_b2 = VectorCreate(a, model->b2->Length);

    s->X_embed = VectorCreate(a, model->E->Cols);
    s->X_hidden = VectorCreate(a, model->W1->Rows);
    s->A = VectorCreate(a, s->X_hidden->Length);
    s->Probs = VectorCreate(a, model->W2->Rows);
    return s;
}

void MLP_Forward(MLP *m, int token_idx, int y, GradState* grad_state) {
    assert(token_idx >= 0 && token_idx < m->E->Rows && "invalid index");

    // embed
    f64* x_embed = grad_state->X_embed->Data;
    for (int c = 0; c < m->E->Cols; c++)  {
        x_embed[c] = m->E->Data[token_idx * m->E->Cols + c];
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
    printf("loss: %.3f\n", loss);
}

void MLP_Backward(MLP* model, int token_id, int y, GradState* grad_state) {

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
    f64 dLdembed[model->E->Cols]; // (n_embed, )
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
    assert(token_id < model->E->Rows && token_id >= 0 && "token id out of bounds");
    for (int c = 0; c < model->E->Cols; c++) {
        grad_state->Grad_E->Data[token_id * model->E->Cols + c] = dLdembed[c];
    }
}



int main(int argc, char *argv[]) {
    srand((uint32_t)123);

    MemoryArena* main_arena = ArenaCreate((size_t)(10 << 20));
    ArenaLog(main_arena);

    const int vocab_size = 27;
    f64 expected_loss = -log(1/(f64)vocab_size);
    MLP* model = MLP_Create(main_arena, vocab_size, 1, 16, 64);
    GradState* grad_state = GradStateCreate(main_arena, model);

    int token_id = 0;
    int y = 1;
    MLP_Forward(model, token_id, y, grad_state);
    MLP_Backward(model, token_id, y, grad_state);

    // TODO sgd
    // TODO zero grad
    
    ArenaLog(main_arena);
    
    printf("expected rnd init loss: %.4f\n", expected_loss);
    printf("size of MLP: %zu\n", sizeof(MLP));


    f64 logits[4] = {-0.02, -0.01, 0.01, 0.02};
    for (int i = 0; i < 4; i++) {
        f64 exp1 = exp(logits[i]);
        f64 loss1 = -log(exp1);
        printf("logit: %.4f exp %.4f: loss: %.4f\n", logits[i], exp1, loss1);
    }

    return 0;
}