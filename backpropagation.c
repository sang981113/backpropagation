#define MAX_LEN 10000000
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double sigmoid(double x);
double sigmoid_prime(double x);

int main()
{
	int pattern_num = 4000;
	int feature_num = 100;

	int n = 100;
	int p = 20;
	int m = 10;
	double lr = 0.5;
	int i, j, k, l;
	int epoch;
	int loop = 0;

	double dot_product = 0;
	double input_unit[101] = { 0, };
	double hidden_in[21] = { 0, };
	double hidden_unit[21] = { 0, };
	double hidden_error[21] = { 0, };
	double output_in[10] = { 0 };
	double output_unit[10] = { 0, };
	double output_error[10] = { 0, };
	double output_error_in[10] = { {0} };
	double target[10] = { 0, };

	//weight initialization
	double v[101][20] = { {0} }; 
	double dv[101][20] = { {0} };
	double nv[101][20] = { {0} };
	double w[21][10] = { {0} };
	double dw[21][10] = { {0} };
	double nw[21][10] = { {0} };

	double** x_train = (double**)malloc(sizeof(double*) * pattern_num); //malloc array x_train[4000][100]
	for (i = 0; i < pattern_num; i++)
	{
		x_train[i] = (double*)malloc(sizeof(double) * feature_num);
	}

	int* y_train = (int*)malloc(sizeof(int) * pattern_num); //malloc array y_train[4000]

	FILE* fp = NULL;
	char* buffer = (char*)calloc(MAX_LEN, sizeof(char));

	if (fopen_s(&fp, "train.txt", "r") == 0) //import data from file
	{	
		for (i = 0; i < pattern_num; i++)
		{
			fscanf_s(fp, "%d\n", &y_train[i]);
			for (j = 0; j < 10; j++)
			{
				for (k = 0; k < 9; k++)
				{
					fscanf_s(fp, "%f ", &x_train[i][j * 10 + k]);
				}
				fscanf_s(fp, "%f\n", &x_train[i][j * 10 + 9]);
			}
		}
		fclose(fp);
	}

	printf("수행할 epoch 수: ");
	scanf("%d", &epoch);

	while (loop < epoch)
	{
		for (l = 0; l < pattern_num; l++)
		{
			//forward propagation
			for (i = 1; i <= n; i++) // insert x_train values into input units
			{
				input_unit[i] = x_train[l][i];
			}
			for (j = 1; j <= p; j++)
			{
				dot_product = 0;
				for (i = 1; i <= n; i++)
				{
					dot_product += input_unit[i] * v[i][j];
				}
				hidden_in[j] = v[0][j] + dot_product;
				hidden_unit[j] = sigmoid(hidden_in[j]);
			}
			for (k = 1; k <= m; k++)
			{
				dot_product = 0;
				for (j = 1; j <= p; j++)
				{
					dot_product += hidden_unit[j] * w[j][k];
				}
				output_in[k] = w[0][k] + dot_product;
				output_unit[k] = sigmoid(output_in[k]);
			}

			//back propagation
			for (i = 1; i <= m; i++)
			{
				output_error[i] = (target[i] - output_unit[i]) * sigmoid_prime(output_in[i]);
				for (j = 0; j <= p; j++)
				{
					dw[j][i] = lr * output_error[j] * hidden_unit[j];
				}
			}
			for (i = 1; i <= p; i++)
			{
				dot_product = 0;
				for (k = 1; k <= m; k++)
				{
					dot_product += output_error[k] * w[i][k];
				}
				output_error_in[i] = dot_product;
				hidden_error[i] = output_error_in[i] * sigmoid_prime(output_in[i]);
				for (j = 0; j <= n; j++)
				{
					dv[j][i] = lr * hidden_error[j] * input_unit[i];
				}
			}
			//update wieghts and biases
			for (i = 1; i <= m; i++)
			{
				for (j = 0; j <= p; j++)
				{
					nw[j][i] = w[j][i] + dw[j][i];
				}
			}
			for (j = 1; j <= p; j++)
			{
				for (i = 0; i <= n; i++)
				{
					nv[j][i] = v[j][i] + dv[j][i];
				}
			}
		}
		loop++;
	}
}

double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double sigmoid_prime(double x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}