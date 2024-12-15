using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Algorithms;
using Range = Microsoft.ML.Probabilistic.Models.Range;
using Microsoft.ML.Probabilistic.Factors;

using System.Globalization;

class SimpleProblemModel
{
    static void Main(string[] args)
    {

        var culture_param = new CultureInfo("en-EN");

        double h = 0.01;
        int size_x = (int) (1 / h);

        Range grid_x = new Range(size_x + 1).Named("Сетка по x");

        VariableArray<double> solution = Variable.Array<double>(grid_x);
        VariableArray<double> precisions = Variable.Array<double>(grid_x);
        VariableArray<double> means = Variable.Array<double>(grid_x);

        //means[grid_x] = Variable.GaussianFromMeanAndPrecision(0, 0.01).ForEach(grid_x);
        //precisions[grid_x] =  Variable.GammaFromMeanAndVariance(1, 1).ForEach(grid_x);
        //precisions[grid_x] = 1;
        solution[grid_x] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(grid_x);

        var precision_link = Variable.GammaFromMeanAndVariance(1, 1);

        using (ForEachBlock x = Variable.ForEach(grid_x))
        {

        var bound1 = x.Index == 0;
        var bound2 = x.Index == size_x;
        using (Variable.If(bound1)) 
        {
            Variable.ConstrainEqual(solution[x.Index], -2);
        }
        using (Variable.IfNot(bound1)) 
        {
            using (Variable.If(bound2)) 
            {
                Variable.ConstrainEqual(solution[x.Index], 3);
            }
            using (Variable.IfNot(bound2)) 
            {
                Variable.ConstrainEqualRandom((solution[x.Index - 1] - 2 * solution[x.Index] + solution[x.Index + 1]) / (h * h), Gaussian.FromMeanAndVariance(0, 0.1));
            }
        }
        }

        InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
        engine.NumberOfIterations = 500000;
        //Console.WriteLine(engine.Infer(precision_link));
        //Console.WriteLine(engine.Infer(precisions));
        var prediction = engine.Infer<DistributionStructArray<Gaussian, double>>(solution);

        using (StreamWriter writer = new StreamWriter("D:/Programming/InverseProblemBN/SimpleProblem_Runs/results.csv"))
        {
        foreach (var g in prediction)
        {
            Console.WriteLine(g);
            writer.WriteLine(g.GetMean().ToString("F18", culture_param));
        }
        }
    
    }
}