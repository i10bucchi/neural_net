#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "MT.h"

/*モデル*/
#define SEED (216) // 552
#define SEED_LIMIT (1000)
#define LEARNING_DATA_NUM (24*3)
#define TEST_NUM (100)
#define TEST_DATA_NUM (24 * 3)

/*ニューラルネット*/
#define LEARN_NUM (1)
#define LIMIT (0.5)
#define NNALPHA (0.005)
#define LOOP_LIMIT (100000)
#define MID_UNIT_NO (3)
#define INPUT_UNIT_NO (1 + MID_UNIT_NO)
#define BATCH_SIZE (4*3)

void initW(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1]);
void make_input_data(int input_data[], int size);
void make_teach_data(int teach_data[], int input_data[], int size);
double calc_forward(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1], double input_nn[INPUT_UNIT_NO], double result_mid[MID_UNIT_NO]);
void setinput(double input_nn[INPUT_UNIT_NO], int input_data, double result_mid[MID_UNIT_NO]);
void calcmidunit(double result[MID_UNIT_NO], double input_nn[INPUT_UNIT_NO], double w[MID_UNIT_NO][INPUT_UNIT_NO + 1]);
double calcoutunit(double result_mid[MID_UNIT_NO], double w[MID_UNIT_NO + 1]);
double sigmoidfunc(double z);
double sigmoiddash(double y);
double tanhfunc(double z);
double tanhdash(double y);
void learning_units(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1], int batch_data_input[BATCH_SIZE], int batch_data_teach[BATCH_SIZE], int input_seq_data[LEARNING_DATA_NUM], double result_mid[MID_UNIT_NO], int times);
void bp_for_outunit(double w_out[MID_UNIT_NO + 1], double result_mid[MID_UNIT_NO], double unit_err, double result, double t);
void bp_for_midunit(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1], double input_nn[INPUT_UNIT_NO], double result_mid[MID_UNIT_NO], double out_unit_err);
double errorsum(double result, int t);
void test(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1], int input_seq_data[TEST_DATA_NUM], int teach_data[TEST_DATA_NUM], double error[TEST_DATA_NUM]);

int main(void) {
    int i, j, l, k, s;
    int learn_data_input[LEARNING_DATA_NUM];
    int learn_data_teach[LEARNING_DATA_NUM];
    int test_data_input[TEST_NUM][TEST_DATA_NUM];
    int test_data_teach[TEST_NUM][TEST_DATA_NUM];
    int batch_data_input[BATCH_SIZE];
    int batch_data_teach[BATCH_SIZE];
    double test_error[TEST_NUM][TEST_DATA_NUM];
    double test_error_average[TEST_DATA_NUM];
    double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1];
    double w_out[MID_UNIT_NO + 1];
    double result_mid[MID_UNIT_NO];
    int min_error_seed = 0;
    double min_error_sum = 100;
    double temp_error_sum;

    for (s = 0; s < SEED_LIMIT; s++) {
        // 乱数初期化
        init_genrand(s);

        // ニューラルネットの重み初期化
        initW(w_mid, w_out);

        // 学習データ作成
        make_input_data(learn_data_input, LEARNING_DATA_NUM);
        make_teach_data(learn_data_teach, learn_data_input, LEARNING_DATA_NUM);

        // テストデータ作成
        for (i = 0; i < TEST_NUM; i++) {
            make_input_data(test_data_input[i], TEST_DATA_NUM);
            make_teach_data(test_data_teach[i], test_data_input[i], TEST_DATA_NUM);
        }

        // バッチサイズ(一定のシーケンスの長さ)ごとに学習
        for (i = 0; i < LEARN_NUM; i++) {
            for (j = 0; j < LEARNING_DATA_NUM; j++) {
                batch_data_input[j % BATCH_SIZE] = learn_data_input[j];
                batch_data_teach[j % BATCH_SIZE] = learn_data_teach[j];

                if (j % BATCH_SIZE == BATCH_SIZE - 1) {
                    learning_units(w_mid, w_out, batch_data_input, batch_data_teach, learn_data_input, result_mid, j - BATCH_SIZE + 1);
                }
            }
        }

        // テスト
        for (i = 0; i < TEST_NUM; i++) {
            test(w_mid, w_out, test_data_input[i], test_data_teach[i], test_error[i]);
        }

        // 結果の表示
        printf("seed:%d error average = { ", s);
        for (i = 0; i < TEST_DATA_NUM; i++) {
            for (j = 0; j < TEST_NUM; j++) {
                test_error_average[i] += test_error[j][i];
            }
            test_error_average[i] /= TEST_NUM;
            printf("%f ", test_error_average[i]);
            if (i % 3 == 1) {
                printf("/");
            }
        }
        printf("}\n");

        temp_error_sum = 0;

        for (i = 0; i < TEST_DATA_NUM; i++) {
            temp_error_sum += test_error_average[i];
        }
        if (temp_error_sum < min_error_sum) {
            min_error_sum = temp_error_sum;
            min_error_seed = s;
        }
        printf("error sum = %f\n", temp_error_sum);
        printf("min error seed = %d\n", min_error_seed);
    }

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

void make_input_data(int input_data[], int size) {
    int i, j;
    int index;
    int sets[4][3] = {
        {0, 0, 0},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 0},
    };

    for (i = 0; i < size; i += 3) {
        index = genrand_int32() % 4;
        for (j = 0; j < 3; j++) {
            input_data[i + j] = sets[index][j];
        }
    }
    
}

void make_teach_data(int teach_data[], int input_data[], int size) {
    int i;

    for (i = 0; i < size - 1; i++) {
        teach_data[i] = input_data[i + 1];
    }
    teach_data[i] = 0;
}

double calc_forward(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1], double input_nn[INPUT_UNIT_NO], double result_mid[MID_UNIT_NO]) {
    double result;

    // 中間層の計算
    calcmidunit(result_mid, input_nn, w_mid);

    // 出力層の計算
    result = calcoutunit(result_mid, w_out);

    return (result);
}

void setinput(double input_nn[INPUT_UNIT_NO], int input_data, double result_mid[MID_UNIT_NO]) {
    int i;

    input_nn[0] = input_data;

    for (i = 0; i < MID_UNIT_NO; i++) {
        input_nn[i + 1] = result_mid[i];
    }
    
}

void calcmidunit(double result[MID_UNIT_NO], double input_nn[INPUT_UNIT_NO], double w[MID_UNIT_NO][INPUT_UNIT_NO + 1]) {
    int i, j;
    double z;
    
    for (i = 0; i < MID_UNIT_NO; i++) {
        z = 0;
        for (j = 0; j < INPUT_UNIT_NO; j++) {
            z += input_nn[j] * w[i][j];
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

void learning_units(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1], int batch_data_input[BATCH_SIZE], int batch_data_teach[BATCH_SIZE], int input_seq_data[LEARNING_DATA_NUM], double result_mid[MID_UNIT_NO], int times) {
    int i, j, k;
    double result;
    double co, cm[MID_UNIT_NO];
    double err = 100.00;
    double input_nn[INPUT_UNIT_NO];
    double t;

    i = 0;
    while ((err) > LIMIT && i < LOOP_LIMIT) {
        err = 0.0;

        // 文脈層初期化
        for (j = 0; j < MID_UNIT_NO; j++) {
            result_mid[j] = sigmoidfunc(0);
        }

        for (j = 0; j < BATCH_SIZE; j++) {
            // 重みを更新したためコンテキスト層を再度求める
            for (k = 0; k < times + j; k++) {
                setinput(input_nn, input_seq_data[k], result_mid);
                calc_forward(w_mid, w_out, input_nn, result_mid);
            }
            // 入力設定
            setinput(input_nn, batch_data_input[j], result_mid);        

            // フォワード計算
            result = calc_forward(w_mid, w_out, input_nn, result_mid);

            // 出力層の学習(OUT -> MID)
            t = batch_data_teach[j];
            co = (result - t) * sigmoiddash(result);
            bp_for_outunit(w_out, result_mid, co, result, t);

            // 中間層の学習(MID -> INPUT)
            bp_for_midunit(w_mid, w_out, input_nn, result_mid, co);

            // 誤差を求める
            err += errorsum(result, t);
        }

        i++;
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

void bp_for_midunit(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1], double input_nn[INPUT_UNIT_NO], double result_mid[MID_UNIT_NO], double out_unit_err) {
    int i, j, k;
    double unit_err;

    // 中間層の学習(MID -> INPUT)
    for (i = 0; i < MID_UNIT_NO; i++) {
        unit_err = sigmoiddash(result_mid[i]) * w_out[i] * out_unit_err;

        for (j = 0; j < INPUT_UNIT_NO; j++) {
            w_mid[i][j] -= NNALPHA * input_nn[j] * unit_err;
        }
        w_mid[i][j] -= NNALPHA * (-1.0) * unit_err;
    }
}

double errorsum(double result, int t) {
    double err;

    err = (result - t) * (result - t);

    return (err);
}

void test(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[MID_UNIT_NO + 1], int input_seq_data[TEST_DATA_NUM], int teach_data[TEST_DATA_NUM], double error[TEST_DATA_NUM]) {
    int i, j;
    double input_nn[INPUT_UNIT_NO];
    double result;
    double result_mid[MID_UNIT_NO];

    for (j = 0; j < MID_UNIT_NO; j++) {
        result_mid[j] = sigmoidfunc(0);
    }

    for (i = 0; i < TEST_DATA_NUM; i++) {
        setinput(input_nn, input_seq_data[i], result_mid);
        result = calc_forward(w_mid, w_out, input_nn, result_mid);
        
        if ((error[i] = (result - teach_data[i])) < 0) {
            error[i] *= -1;
        }
    }
}