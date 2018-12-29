#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "MT.h"

/*モデル*/
#define SEED (48129)
#define LEARNING_DATA_NUM (4*3)
#define TEST_DATA_NUM (4 * 3)

/*ニューラルネット*/
#define LEARN_NUM (100)
#define LIMIT (0.01)
#define NNALPHA (0.01)
#define LOOP_LIMIT (100000)
#define MID_UNIT_NO (2)
#define INPUT_UNIT_NO (2)
#define BATCH_SIZE (LEARNING_DATA_NUM)

void initW(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1]);
void printW(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1]);
void make_input_data(int input_data[]);
double calc_forward(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1], double input[INPUT_UNIT_NO], double result_mid[MID_UNIT_NO]);
void setinput(double input[INPUT_UNIT_NO], int input_data, double result_mid[MID_UNIT_NO]);
void calcmidunit(double result[MID_UNIT_NO], double input[INPUT_UNIT_NO], double w[MID_UNIT_NO][INPUT_UNIT_NO + 1]);
double calcoutunit(double result_mid[MID_UNIT_NO], double w[MID_UNIT_NO + 1]);
double sigmoidfunc(double z);
double sigmoiddash(double y);
double tanhfunc(double z);
double tanhdash(double y);
void learning_units(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1], int input_data[BATCH_SIZE], double result_mid[MID_UNIT_NO]);
void bp_for_outunit(double w_out[MID_UNIT_NO + 1], double result_mid[MID_UNIT_NO], double unit_err, double result, double t);
void bp_for_midunit(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1], double input[INPUT_UNIT_NO], double result_mid[MID_UNIT_NO], double out_unit_err);
double errorsum(double result, int t);
void test(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1], int input_seq_data[TEST_DATA_NUM]);

int main(void) {
    int i, j, l;
    int data_input[LEARNING_DATA_NUM];
    double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1];
    double w_out[MID_UNIT_NO + 1];
    double result_mid[MID_UNIT_NO];
    double temp_result;
    double temp_input[INPUT_UNIT_NO];

    init_genrand(SEED);

    // ニューラルネットの重み初期化
    initW(w_mid, w_out);
    // 初期設定の表示
    printW(w_mid, w_out);

    // データ作成
    make_input_data(data_input);

    for (l = 0; l < LEARN_NUM; l++) {
        // 学習
        learning_units(w_mid, w_out, data_input, result_mid);
            
        printf("learning: %d / %d\n", l, LEARN_NUM);
    }

    // テスト
    test(w_mid, w_out, data_input);

    printW(w_mid, w_out);

    return (0);
}

void initW(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1]) {
    int i, j;
    for (i = 0; i < MID_UNIT_NO; i++) {
        for (j = 0; j < INPUT_UNIT_NO + 1; j++) {
            w_mid[i][j] = genrand_real1() * 2 - 1;
        }
    }

    for (j = 0; j < MID_UNIT_NO + 1; j++) {
        w_out[j] = genrand_real1() * 2 - 1;
    }
}

void printW(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1]) {
    int i, j;

    printf("###################### printw #########################\n");
    printf("w_mid----------------------\n");
    for (i = 0; i < MID_UNIT_NO; i++) {
        for (j = 0; j < INPUT_UNIT_NO + 1; j++) {
            printf("%d -> %d: %lf\n", j, i, w_mid[i][j]);
        }
    }
    printf("w_out----------------------\n");
    for (j = 0; j < MID_UNIT_NO + 1; j++) {
        printf("%d -> %d: %lf\n", j, i, w_out[j]);
    }
}

void make_input_data(int input_data[]) {
    int i, j;
    int index;
    int sets[4][3] = {
        {0, 0, 0},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 0},
    };

    printf("make data\n");

    for (i = 0; i < LEARNING_DATA_NUM; i+=3) {
        for (j = 0; j < 3; j++) {
            input_data[i + j] = sets[i / 3][j];
            printf("%d", input_data[i + j]);
        }
        printf("/");
    }
    printf("\n");
    
}

double calc_forward(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1], double input[INPUT_UNIT_NO], double result_mid[MID_UNIT_NO]) {
    double result;

    // 中間層の計算
    calcmidunit(result_mid, input, w_mid);

    // 出力層の計算
    result = calcoutunit(result_mid, w_out);

    return (result);
}

void calcmidunit(double result[MID_UNIT_NO], double input[INPUT_UNIT_NO], double w[MID_UNIT_NO][INPUT_UNIT_NO + 1]) {
    int i, j;
    double z;
    
    for (i = 0; i < MID_UNIT_NO; i++) {
        z = 0;
        for (j = 0; j < INPUT_UNIT_NO; j++) {
            z += input[j] * w[i][j];
        }
        z += (-1) * w[i][j];

        result[i] = sigmoidfunc(z);
    }
    
}

double calcoutunit(double result_mid[MID_UNIT_NO], double w[MID_UNIT_NO + 1]) {
    int i, j;
    double z;
    double result;
    
    z = 0;
    for (j = 0; j < MID_UNIT_NO; j++) {
        z += result_mid[j] * w[j];
    }
    z += (-1) * w[j];

    result = sigmoidfunc(z);

    return (result);
}

double sigmoidfunc(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double sigmoiddash(double y) {
    return (y * (1 - y));
}

double tanhfunc(double z) {
    return tanh(z);
}

double tanhdash(double y) {
    return (4 / ((exp(y) + exp(-y)) * (exp(y) + exp(-y))));
}

void learning_units(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1], int input_data[BATCH_SIZE], double result_mid[MID_UNIT_NO]) {
    int i, j, k;
    double result;
    double co, cm[MID_UNIT_NO];
    double err = 100.00;
    double input[INPUT_UNIT_NO];
    double t;

    j = 0;
    while ((err) > LIMIT && j < LOOP_LIMIT) {
        err = 0.0;

        for (i = 0; i < BATCH_SIZE; i += 3) {

            // 入力設定
            input[0] = input_data[i];
            input[1] = input_data[i + 1];

            // フォワード計算
            result = calc_forward(w_mid, w_out, input, result_mid);

            // 出力層の学習(OUT -> MID)
            t = input_data[i + 2];
            // 出力層ユニット誤差取得
            co = (result - t) * sigmoiddash(result);

            bp_for_outunit(w_out, result_mid, co, result, t);

            // 中間層の学習(MID -> INPUT)
            bp_for_midunit(w_mid, w_out, input, result_mid, co);

            // 誤差を求める
            err += errorsum(result, t);
        }
        printf("unit learning err = %lf\n", err);

        j++;
    }
}

void bp_for_outunit(double w_out[MID_UNIT_NO + 1], double result_mid[MID_UNIT_NO], double unit_err, double result, double t) {
    int i, j;

    // 出力層の学習(OUT -> MID)
    for (j = 0; j < MID_UNIT_NO; j++) {
        w_out[j] -= NNALPHA * result_mid[j] * unit_err;
    }
    w_out[j] -= NNALPHA * (-1.0) * unit_err;
}

void bp_for_midunit(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1], double input[INPUT_UNIT_NO], double result_mid[MID_UNIT_NO], double out_unit_err) {
    int i, j, k;
    double unit_err;

    // 中間層の学習(MID -> INPUT)
    for (i = 0; i < MID_UNIT_NO; i++) {
        unit_err = sigmoiddash(result_mid[i]) * w_out[i] * out_unit_err;

        for (j = 0; j < INPUT_UNIT_NO; j++) {
            w_mid[i][j] -= NNALPHA * input[j] * unit_err;
        }
        w_mid[i][j] -= NNALPHA * (-1.0) * unit_err;
    }
}

double errorsum(double result, int t) {
    double err;

    err = (result - t) * (result - t);

    return (err);
}

void test(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1], int input_seq_data[TEST_DATA_NUM]) {
    int i, j;
    double input[INPUT_UNIT_NO];
    double result;
    double result_mid[MID_UNIT_NO];

    for (i = 0; i < LEARNING_DATA_NUM; i += 3) {
        input[0] = input_seq_data[i];
        input[1] = input_seq_data[i + 1];
        printf("input = { ");
        for (j = 0; j < INPUT_UNIT_NO; j++) {
            printf("%f ", input[j]);
        }
        printf("}\n");
        result = calc_forward(w_mid, w_out, input, result_mid);
        printf("%d : %lf\n", input_seq_data[i + 2], result);
    }
}