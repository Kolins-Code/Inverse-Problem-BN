using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Algorithms;
using Range = Microsoft.ML.Probabilistic.Models.Range;

using System.IO;
using System.Globalization;
using System.Diagnostics;


const string NAME = "data.csv";

double dt = 0.01;
double dx = 0.1;
double end_x = Math.PI;
double end_t = 1;
int size_x = (int) (end_x / dx);
int size_t = (int) (end_t / dt);
Console.WriteLine(size_t);

double gamma = dt / (dx * dx);
Console.WriteLine(gamma);

var reader = new StreamReader(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName + '\\' + NAME);

double[] end_function_obs = new double[size_x + 1];

int k = 0;
while (!reader.EndOfStream && k < size_x + 1)
{
    var line = reader.ReadLine();
    end_function_obs[k] = Convert.ToDouble(line, new CultureInfo("en-EN"));
    //Console.WriteLine(end_function_obs[k]);
    k++;
}
reader.Close();


size_t = 100;
Range grid_x = new Range(size_x + 1).Named("Сетка по x");

VariableArray<double>[] solution = new VariableArray<double>[size_t + 1];
for (int i = 0; i < solution.Length; i++)
{
    solution[i] = Variable.Array<double>(grid_x).Named("Решение прямой задачи, слой " + i);
}
solution[0][grid_x] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(grid_x);
//VariableArray2D<double> solution = Variable.Array<double>(grid_x, grid_tt).Named("Решение прямой задачи");
/*VariableArray<double> start_function = Variable.Array<double>(grid_x).Named("Начальное условие");
start_function[grid_x] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(grid_x);
VariableArray<double> end_function = Variable.Array<double>(grid_x).Named("Наблюдение");*/

/*for (int i = 0; i < size_t - 1; i++)
{
    solution[0, i] = 0;
    solution[size_x, i] = 0;
}

for (int i = 0; i < size_x - 1; i++)
{
    solution[i + 1, 0] = (1 - 2 * gamma) * start_function[i + 1] + gamma * (start_function[i] + start_function[i + 2]);
}

for (int n = 0; n < size_t - 2; n++)
{
    for (int i = 0; i < size_x - 1; i++)
    {
        solution[i + 1, n + 1] = (1 - 2 * gamma) * solution[i + 1, n] + gamma * (solution[i, n] + solution[i + 2, n]);
    }
}

for (int i = 0; i < size_x - 1; i++)
{
    end_function[i + 1] = (1 - 2 * gamma) * solution[i + 1, size_t - 2] + gamma * (solution[i, size_t - 2] + solution[i + 2, size_t - 2]);
}*/
for (int t = 1; t <= size_t; t++)
{
    using (ForEachBlock x = Variable.ForEach(grid_x))
    {
        using (Variable.If(x.Index == 0)) 
        {
            solution[t][x.Index] = Variable.GaussianFromMeanAndPrecision(0, Double.PositiveInfinity);
        }
        using (Variable.IfNot(x.Index == 0)) 
        {
            using (Variable.If(x.Index == size_x)) 
            {
                solution[t][x.Index] = Variable.GaussianFromMeanAndPrecision(0, Double.PositiveInfinity);
            }
            using (Variable.IfNot(x.Index == size_x)) 
            {
                solution[t][x.Index] = ((1 - gamma) * solution[t - 1][x.Index] + 0.5 * gamma * (solution[t][x.Index - 1] + solution[t][x.Index + 1]) +  0.5 * gamma * (solution[t - 1][x.Index - 1] + solution[t - 1][x.Index + 1])) / (1 + gamma);
            }
            
        }
    }  
}


solution[size_t].ObservedValue = end_function_obs;


InferenceEngine engine = new InferenceEngine();
//engine.Compiler.CompilerChoice = CompilerChoice.Roslyn;
//engine.SaveFactorGraphToFolder = "graphs";
engine.NumberOfIterations = 1000;


double[] prediction = new double[size_x + 1];

Console.WriteLine(engine.Infer<DistributionStructArray<Gaussian, double>>(solution[0]));

