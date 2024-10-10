﻿using Microsoft.ML.Probabilistic.Distributions;
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


Range grid_x = new Range(size_x + 1).Named("Сетка по x");
Range grid_t = new Range(size_t).Named("Сетка по t");
Range grid_tt = new Range(size_t - 1).Named("");

VariableArray2D<double> solution = Variable.Array<double>(grid_x, grid_tt).Named("Решение прямой задачи");
VariableArray<double> start_function = Variable.Array<double>(grid_x).Named("Начальное условие");
start_function[grid_x] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(grid_x);
VariableArray<double> end_function = Variable.Array<double>(grid_x).Named("Наблюдение");

Range iterations_x = new Range(size_x - 1).Named("Итерации по внутренним точкам x");
Range iterations_t = new Range(size_t - 2).Named("Итерации по внутренним точкам t");

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
using (ForEachBlock t = Variable.ForEach(grid_t))
{
    using (ForEachBlock x = Variable.ForEach(grid_x))
    {
        using (Variable.If(x.Index == 0)) 
        {
            using (Variable.IfNot(t.Index == size_t - 1)) 
            {
                solution[x.Index, t.Index] = 0;
            }
        }
        using (Variable.IfNot(x.Index == 0)) 
        {
            using (Variable.If(x.Index == size_x)) 
            {
                using (Variable.IfNot(t.Index == size_t - 1)) 
                {
                    solution[x.Index, t.Index] = 0;
                }
            }
            using (Variable.IfNot(x.Index == size_x)) 
            {
                /*using (Variable.If(t.Index == 0)) 
                {
                    solution[x.Index, 0] =  0/*(1 - 2 * gamma) * start_function[x.Index] + gamma * (start_function[x.Index - 1] + start_function[x.Index + 1]);
                }*/
                using (Variable.IfNot(t.Index == 0)) 
                {
                    using (Variable.If(t.Index == size_t - 1)) 
                    {
                        end_function[x.Index] = (1 - 2 * gamma) * solution[x.Index, t.Index - 1] + gamma * (solution[x.Index - 1, t.Index - 1] + solution[x.Index + 1, t.Index - 1]);
                    }
                    using (Variable.IfNot(t.Index == size_t - 1)) 
                    {
                        solution[x.Index, t.Index] = (1 - 2 * gamma) * solution[x.Index, t.Index - 1] + gamma * (solution[x.Index - 1, t.Index - 1] + solution[x.Index + 1, t.Index - 1]);
                    }
                    
                }
            }
            
        }
    }  
}


end_function.ObservedValue = end_function_obs;


InferenceEngine engine = new InferenceEngine();
//engine.Compiler.CompilerChoice = CompilerChoice.Roslyn;
//engine.SaveFactorGraphToFolder = "graphs";
engine.NumberOfIterations = 50;


double[] prediction = new double[size_x + 1];

Console.WriteLine(engine.Infer<DistributionStructArray<Gaussian, double>>(solution));

