#define MAX_LEN 10000000
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float sigmoid(float x);
float sigmoid_prime(float x);
int* get_target_vector(int target);
float random_float(void);

int main()
{
	int pattern_num = 4000;
	int feature_num = 100;

	int n = 100;
	int p = 20;
	int m = 10;
	float lr = 0.5;
	int i, j, k, l;
	int epoch;
	int loop = 0;
	int score = 0;
	int target_num = 0;
	float target_score = 0;
	float dot_product = 0;
	float input_unit[101];
	float hidden_in[21];
	float hidden_unit[21];
	float hidden_error[21];
	float output_error_in[21];
	float output_in[11];
	float output_unit[11];
	float output_error[11];
	for (i = 0; i <= n; i++)
	{
		input_unit[i] = random_float();
	}
	for (i = 0; i <= p; i++)
	{
		hidden_in[i] = random_float();
		hidden_unit[i] = random_float();
		hidden_error[i] = random_float();
		output_error_in[i] = random_float();
	}
	for (i = 0; i <= m; i++)
	{
		output_in[i] = random_float();
		output_unit[i] = random_float();
		output_error[i] = random_float();
	}

	int *target;

	float v[101][21];
	float dv[101][21];
	float nv[101][21];
	float w[21][11];
	float dw[21][11];
	float nw[21][11];
	//weight initialization
	for (i = 0; i <= n; i++)
	{
		for (j = 0; j <= p; j++)
		{
			v[i][j] = random_float();
			dv[i][j] = random_float();
			nv[i][j] = random_float();
		}
	}
	for (j = 0; j <= p; j++)
	{
		for (k = 0; k <= m; k++)
		{
			w[j][k] = random_float();
			dw[j][k] = random_float();
			nw[j][k] = random_float();
		}
	}

	float** x_train = (float**)malloc(sizeof(float*) * pattern_num); //malloc array x_train[4000][100]
	for (i = 0; i < pattern_num; i++)
	{
		x_train[i] = (float*)malloc(sizeof(float) * feature_num);
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
	scanf_s("%d", &epoch);

	while (loop < epoch)
	{
		for (l = 0; l < pattern_num; l++)
		{
			int this_pattern = l * 400 / 4000 + (l * 400) % 4000;
			for (i = 1; i <= n; i++)
			{
				for (j = 1; j <= p; j++)
				{
					v[i][j] = nv[i][j];
				}
			}
			for (j = 1; j <= p; j++)
			{
				for (k = 1; k <= m; k++)
				{
					w[j][k] = nw[j][k];
				}
			}
			
			//forward propagation
			for (i = 1; i <= n; i++) // insert x_train values into input units
			{
				input_unit[i] = x_train[this_pattern][i-1];
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

			//validation
			target_score = 0;
			for (i = 1; i <= m; i++)
			{
				if (target_score < output_unit[i])
				{
					target_score = output_unit[i];
					target_num = i;
				}
			}
			if (target_num - 1 == y_train[this_pattern])
			{
				score++;
			}
			target = get_target_vector(y_train[this_pattern]);

			//back propagation
			for (i = 1; i <= m; i++)
			{
				output_error[i] = (target[i] - output_unit[i]) * sigmoid_prime(output_in[i]);
				for (j = 0; j <= p; j++)
				{
					dw[j][i] = lr * output_error[i] * hidden_unit[j];
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
				hidden_error[i] = output_error_in[i] * sigmoid_prime(hidden_in[i]);
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
		printf("train acc: %f\n", (float) score / pattern_num);
		score = 0;
		loop++;
	}
}

float sigmoid(float x)
{
	return 1 / (1 + exp(-x));
}

float sigmoid_prime(float x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}

int* get_target_vector(int target)
{
	int* target_vector = (int*)malloc(sizeof(int) * 11);
	for (int i = 1; i <= 10; i++)
	{
		if (target == i - 1)
		{
			target_vector[i] = 1;
		}
		else
		{
			target_vector[i] = 0;
		}
	}
	return target_vector;
}

float random_float(void)
{
	return (float)rand() / 32768;
}