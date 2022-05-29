#define MAX_LEN 10000000
#include <stdio.h>
#include <stdlib.h>

int main()
{
	int pattern_num = 4000;
	int feature_num = 100;

	int n = 100;
	int p = 20;
	int m = 10;
	int i, j, k, l;
	int epoch;
	int loop = 0;

	float input_unit[100] = { 0, };
	float hidden_unit[20] = { 0, };
	float output_unit[10] = { 0, };

	float v[100][20] = {{0}}; //weight initialization
	float w[20][10] = {{0}};

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
	scanf("%d", &epoch);

	while (loop < epoch)
	{
		for (l = 0; l < pattern_num; l++)
		{
			for (i = 1; i <= n; i++) // insert x_train values into input units
			{
				input_unit[i] = x_train[l][i];
			}
			for (j = 1; j <= p; j++)
			{

			}
			for (k = 1; k <= m; k++)
			{

			}
			for (i = 1; i <= m; i++)
			{

			}
			for (i = 1; i <= p; i++)
			{
				for (j = 0; j <= n; j++)
				{

				}
			}
			for (i = 1; i <= m; i++)
			{
				for (j = 0; j <= p; j++)
				{

				}
			}
			for (j = 1; j <= p; j++)
			{
				for (i = 0; i <= n; i++)
				{

				}
			}
		}
		loop++;
	}
}