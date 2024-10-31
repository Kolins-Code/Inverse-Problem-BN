using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Algorithms;
using Range = Microsoft.ML.Probabilistic.Models.Range;

using System.IO;
using System.Globalization;
using System.Diagnostics;


class Model
{
    static void Main(string[] args)
    {
        var culture_param = new CultureInfo("en-EN");

        double dt = Convert.ToDouble(args[0], culture_param);
        double dx = Convert.ToDouble(args[1], culture_param);
        double end_x = Convert.ToDouble(args[3], culture_param);
        double end_t = Convert.ToDouble(args[2], culture_param);

        int size_x = (int) (end_x / dx);
        int size_t = (int) (end_t / dt);
        //Console.WriteLine(size_t);

        double gamma = dt / (dx * dx);
        //Console.WriteLine(gamma);

        var reader = new StreamReader(args[4]);

        double[] end_function_obs = new double[size_x + 1];

        int k = 0;
        while (!reader.EndOfStream && k < size_x + 1)
        {
            var line = reader.ReadLine();
            end_function_obs[k] = Convert.ToDouble(line, culture_param);
            //Console.WriteLine(end_function_obs[k]);
            k++;
        }
        reader.Close();


        Range grid_x = new Range(size_x + 1).Named("Сетка по x");

        VariableArray<double>[] solution = new VariableArray<double>[size_t + 1];
        VariableArray<double> means = Variable.Array<double>(grid_x).Named("Средние");
        VariableArray<double> precisions = Variable.Array<double>(grid_x).Named("Обратные дисперсии");
        means[grid_x] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(grid_x);
        precisions[grid_x] = Variable.GammaFromShapeAndScale(1, 1).ForEach(grid_x);

        for (int i = 0; i < solution.Length; i++)
        {
            solution[i] = Variable.Array<double>(grid_x).Named("Решение прямой задачи, слой " + i);
        }
        for (int i = 1; i < size_x; i++)
        {
            solution[0][i] = Variable.GaussianFromMeanAndPrecision(0, 0.01);
            //solution[0][i] = Variable.GaussianFromMeanAndPrecision(means[i], precisions[i]);
        }
        solution[0][0] = Variable.GaussianFromMeanAndPrecision(0, Double.PositiveInfinity);
        solution[0][size_x] = Variable.GaussianFromMeanAndPrecision(0, Double.PositiveInfinity);

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
                        solution[t][x.Index] = (solution[t - 1][x.Index] + gamma * (solution[t][x.Index - 1] + solution[t][x.Index + 1])) / (1 + 2 * gamma) ;
                    }
                    
                }
            }  
        }


        solution[size_t].ObservedValue = end_function_obs;


        InferenceEngine engine = new InferenceEngine();
        engine.SaveFactorGraphToFolder = "graphs";
        engine.NumberOfIterations = 200;

        var prediction = engine.Infer<DistributionStructArray<Gaussian, double>>(solution[0]);
        //Console.WriteLine(prediction);
        double[] output_function = new double[size_x + 1];
        int index = 0;
        foreach (var point in prediction)
        {
            output_function[index++] = point.GetMean();
        }
        //Console.WriteLine(engine.Infer<DistributionStructArray<Gaussian, double>>(means));
        //Console.WriteLine(engine.Infer<DistributionStructArray<Gamma, double>>(precisions));

        using (StreamWriter writer = new StreamWriter(args[5]))
        {
            foreach (double d in output_function)
                writer.WriteLine(d.ToString("F18", culture_param));
        }
    }
}


