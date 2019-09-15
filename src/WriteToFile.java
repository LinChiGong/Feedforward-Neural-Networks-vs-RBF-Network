/**
 * This class uses Feedforward Neural Network and RBF Network to solve 5
 * classification problems specific to this project. Performance of each task
 * is obtained from 5-fold cross validation. For each dataset, the trained
 * Feedforward Neural Network model, the trained RBF Network model and their
 * prediction results from all 5 folds are written to dataset-specific files.
 * Additionally, the classification accuracies from all tasks are written to a
 * single output file called "Results.txt" for comparison
 * 
 * @author Winston Lin
 */

import java.io.*;
import java.util.*;

public class WriteToFile 
{
    public static void main(String[] args) throws IOException 
    {
        // 5 datasets used in this project
        String[] datasets = {"soybean-small.data", "house-votes-84.data", 
                    "breast-cancer-wisconsin.data", "iris.data", "glass.data"};
        // Write to the file that contains results from all datasets
        PrintWriter fout = new PrintWriter(
                new BufferedWriter(new FileWriter("Results.txt", false)));
        fout.println("Perform classification using Feedforward Neural Network "
                + "and Radial Basis Network on 5 datasets from the "
                + "UCI Machine Learning repository. Performances shown here "
                + "are average classification accuracies obtained from 5-fold "
                + "cross validation.");
        fout.println();
                
        for (int d = 0; d < datasets.length; d++)
        {
            // Write the models and results of each task to corresponding file
            PrintWriter feach = new PrintWriter(
                          new BufferedWriter(
                          new FileWriter(datasets[d] + "-output.txt", false)));
            
            fout.println(datasets[d].substring(datasets[d].lastIndexOf('/')+1,
                    datasets[d].lastIndexOf('.')));
            fout.println("-----------------------");
            
            // Process and split data into training set and test set
            ArrayList<String[]> records = new ArrayList<String[]>();
            ArrayList<ArrayList<String[]>> partitions = 
                    new ArrayList<ArrayList<String[]>>();
            ETL etl = new ETL();
            records = etl.readCSV(datasets[d]);
            partitions = etl.split(records, true);
            feach.println("Dataset: " 
                    + datasets[d].substring(datasets[d].lastIndexOf('/') + 1,
                      datasets[d].lastIndexOf('.')) 
                    + " from UCI Machine Learning repository");
            
            for (int a = 0; a < 2; a++)
            { 
                if (a == 0)
                {
                    feach.println();
                    feach.println(" ---------------------------- ");
                    feach.println("| FEEDFORWARD NEURAL NETWORK |");
                    feach.println(" ---------------------------- ");
                }
                else
                {
                    feach.println();
                    feach.println(" ------------------------------- ");
                    feach.println("| RADIAL BASIS FUNCTION NETWORK |");
                    feach.println(" ------------------------------- ");
                }
    
                // Perform 5-fold cross validation
                double[] performances0 = new double[5];
                double[] performances1 = new double[5];
                double[] performances2 = new double[5];
                double[] performances = new double[5];
                for (int k = 0; k < 5; k++)
                {
                    ArrayList<String[]> Xtrain = new ArrayList<String[]>();
                    ArrayList<String> ytrain = new ArrayList<String>();
                    ArrayList<String[]> Xtest = new ArrayList<String[]>();
                    ArrayList<String> ytest = new ArrayList<String>();
                    
                    for (int i = 0; i < partitions.size(); i++)
                    {
                        if (i == k)
                        {
                            for (String[] test : partitions.get(i))
                            {
                                String[] X = new String[test.length - 1];
                                for (int j = 0; j < X.length; j++)
                                {
                                    X[j] = test[j];
                                }
                                Xtest.add(X);
                                ytest.add(test[test.length - 1]);
                            }
                        }
                        else
                        {
                            for (String[] train : partitions.get(i))
                            {
                                String[] X = new String[train.length - 1];
                                for (int j = 0; j < X.length; j++)
                                {
                                    X[j] = train[j];
                                }
                                Xtrain.add(X);
                                ytrain.add(train[train.length - 1]);
                            }
                        }
                    }  
                    
                    if (a == 0)
                    {// Perform Feedforward Network and store the performances
                        for (int h = 0; h < 3; h++)
                        {
                            feach.println();
                            feach.println(h + " HIDDEN LAYERS - CROSS "
                                    + "VALIDATION FOLD " + (k+1));
                            feach.println("==========================="
                                    + "==============");
                            
                            ArrayList<String> ypred = new ArrayList<String>();
                            
                            double threshold = 0.95;
                            int iteration = 0;
                            double eta = 0.2;
                            if (h == 0)
                            {
                                iteration = 100;
                            }
                            else if (h == 1)
                            {
                                iteration = 1000;
                            }
                            else
                            {
                                iteration = 10000;
                            }
                            if (etl.fileName.equals("glass"))
                            {
                                threshold = 0.65;
                                if (h == 2)
                                {
                                    eta = 0.01;
                                }
                            }
                            FeedForwardNetwork ffn = new FeedForwardNetwork(
                                    h, -1, threshold, iteration, eta);
                            ffn.fit(Xtrain, ytrain);
                            ypred = ffn.predict(Xtest);
                            
                            // Write the model and predictions from each fold
                            feach.println();
                            feach.println("Weights of each node");
                            feach.println("--------------------");
                            
                            for (HiddenNode[] hiddenLayer : ffn.hiddenLayers)
                            {
                                feach.println("Hidden layer: ");
                                int i = 1;
                                for (HiddenNode hNode : hiddenLayer)
                                {
                                    feach.print("Hidden node " + i + ": [ ");
                                    for (double weight : hNode.weights)
                                    {
                                        feach.print(weight + " ");
                                    }
                                    feach.println("]");
                                    i++;
                                }
                            }
                            feach.println("Output layer: ");
                            int j = 1;
                            for (OutputNode oNode : ffn.outputLayer)
                            {
                                feach.print("Output node " + j + ": [ ");
                                for (double weight : oNode.weights)
                                {
                                    feach.print(weight + " ");
                                }
                                feach.println("]");
                                j++;
                            }
                            feach.println();
                            
                            feach.println("Classification results");
                            feach.println("----------------------");
                            feach.println();
                            feach.println("\tACTUAL" + "\t \t" + "PREDICTED");
                            feach.println("\t------" + "\t \t" + "---------");
                            
                            double performance = 0;
                            for (int i = 0; i < ypred.size(); i++)
                            {
                                feach.println("\t" + ytest.get(i) 
                                                   + "\t|\t" + ypred.get(i));
                                if (ytest.get(i).equals(ypred.get(i)))
                                {
                                    performance++;
                                }
                            }
                            performance = performance / ytest.size();
                            if (h == 0)
                            {
                                performances0[k] = performance;
                            }
                            else if (h == 1)
                            {
                                performances1[k] = performance;
                            }
                            else
                            {
                                performances2[k] = performance;
                            }
                            
                            feach.println("--------------------------------");
                            feach.println("Classification accuracy: " 
                            + Math.round(performance * 10000.0) / 100.0 + "%");
                            feach.println();
                        }
                    }
                    else
                    {
                        feach.println();
                        feach.println("CROSS VALIDATION FOLD " + (k+1));
                        feach.println("=======================");
                        
                        // Perform RBF Network and store the performances
                        ArrayList<String> ypred = new ArrayList<String>();
                        
                        int numHidden = 6;
                        double threshold = 0.95;
                        int iteration = 1000;
                        double eta = 20;
                        if (etl.fileName.equals("iris"))
                        {
                            numHidden = 15;
                        }
                        if (etl.fileName.equals("glass"))
                        {
                            numHidden = 75;
                            eta = 1;
                        }
                        
                        RBFNetwork rbf = new RBFNetwork(numHidden, threshold, 
                                iteration, eta);
                        rbf.fit(Xtrain, ytrain);
                        ypred = rbf.predict(Xtest);
                        
                        // Write the model and predictions from each fold
                        feach.println();
                        feach.println("Weights of each node");
                        feach.println("--------------------");
                        
                        int j = 1;
                        for (RBFNetwork.Node node : rbf.nodes)
                        {
                            feach.print("Hidden node " + j + ": [ ");
                            for (double weight : node.weights)
                            {
                                feach.print(weight + " ");
                            }
                            feach.println("]");
                            j++;
                        }
                        feach.println();
                        
                        feach.println("Classification results");
                        feach.println("----------------------");
                        feach.println();
                        feach.println("\tACTUAL" + "\t \t" + "PREDICTED");
                        feach.println("\t------" + "\t \t" + "---------");
                        
                        double performance = 0;
                        for (int i = 0; i < ypred.size(); i++)
                        {
                            feach.println("\t" + ytest.get(i) 
                                               + "\t|\t" + ypred.get(i));
                            if (ytest.get(i).equals(ypred.get(i)))
                            {
                                performance++;
                            }
                        }
                        performance = performance / ytest.size();
                        performances[k] = performance;
                        feach.println("--------------------------------");
                        feach.println("Classification accuracy: " 
                        + Math.round(performance * 10000.0) / 100.0 + "%");
                        feach.println();
                    }
                }
                
                // Output average performances of all tasks to "Results.txt"
                if (a == 0)
                {
                    double average = 0;
                    for (double performance : performances0)
                    {
                        average += performance;
                    }
                    average /= 5.0;
                    fout.println("Feedforward Network - No hidden layer: " 
                                + Math.round(average * 10000.0) / 100.0 + "%");
                    
                    average = 0;
                    for (double performance : performances1)
                    {
                        average += performance;
                    }
                    average /= 5.0;
                    fout.println("Feedforward Network - One hidden layer: " 
                                + Math.round(average * 10000.0) / 100.0 + "%");
                    
                    average = 0;
                    for (double performance : performances2)
                    {
                        average += performance;
                    }
                    average /= 5.0;
                    fout.println("Feedforward Network - Two hidden layers: " 
                                + Math.round(average * 10000.0) / 100.0 + "%");
                }
                else
                {
                    double average = 0;
                    for (double performance : performances)
                    {
                        average += performance;
                    }
                    average /= 5.0;
                    fout.println("Radial Basis Function Network: " 
                                + Math.round(average * 10000.0) / 100.0 + "%");
                    fout.println();
                }
            }
            
            feach.close();
        }
        
        fout.close();
    }
}
