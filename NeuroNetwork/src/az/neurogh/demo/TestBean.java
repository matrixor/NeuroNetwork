package az.neurogh.demo;

import java.util.ArrayList;
import java.util.Arrays;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.IterativeLearning;
import org.neuroph.nnet.learning.LMS;
import org.neuroph.util.TransferFunctionType;

import az.neurogh.util.DataProcessBean;

public class TestBean implements LearningEventListener {

	public void testSimplePerceptron_And(){
		DataSet trainingSet = new DataSet(2, 1);
        trainingSet.addRow(new DataSetRow(new double[]{0, 0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{0, 1}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 1}, new double[]{1}));
        
        SimplePerceptron myPerceptron = new SimplePerceptron(2);

        myPerceptron.getLearningRule().addListener(this);
        
        System.out.println("Training neural network SimplePerceptron_And...");
        myPerceptron.learn(trainingSet);
        
        System.out.println("Testing trained neural network SimplePerceptron_And");
        testNeuralNetwork(myPerceptron, trainingSet);
	}
	
	public void testSimplePerceptron_Or(){
		DataSet trainingSet = new DataSet(2, 1);
        trainingSet.addRow(new DataSetRow(new double[]{0, 0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{0, 1}, new double[]{1}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 0}, new double[]{1}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 1}, new double[]{1}));
        
        SimplePerceptron myPerceptron = new SimplePerceptron(2);

        myPerceptron.getLearningRule().addListener(this);
        
        System.out.println("Training neural network SimplePerceptron_Or...");
        myPerceptron.learn(trainingSet);
        
        System.out.println("Testing trained neural network SimplePerceptron_Or");
        testNeuralNetwork(myPerceptron, trainingSet);
	}
	
	public void testMultiPerceptron_AndOutputNoLearn(){
		DataSet trainingSet = new DataSet(2,1);
        trainingSet.addRow(new DataSetRow(new double[]{0, 0}, new double[]{Double.NaN}));
        trainingSet.addRow(new DataSetRow(new double[]{0, 1}, new double[]{Double.NaN}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 0}, new double[]{Double.NaN}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 1}, new double[]{Double.NaN}));
        
        MultiPerceptron_AndOutputNoLearn myPerceptron = new MultiPerceptron_AndOutputNoLearn(TransferFunctionType.STEP,2,2,1);

        for(DataSetRow testSetRow : trainingSet.getRows()) {
        	myPerceptron.setInput(testSetRow.getInput());
        	myPerceptron.calculate();
            double[] networkOutput = myPerceptron.getOutput();

            System.out.print("Input: " + Arrays.toString( testSetRow.getInput() ) );
            System.out.println(" Output: " + Arrays.toString( networkOutput) );
        }
	}
	
	public void testBinOutputBPNN(){
		DataSet trainingSet = new DataSet(2,1);
		trainingSet.addRow(new DataSetRow(new double[]{0, 0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{0, 1}, new double[]{1}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 0}, new double[]{1}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 1}, new double[]{0}));
        
        BinOutputBPNN myPerceptron = new BinOutputBPNN(TransferFunctionType.SIGMOID,2,4,1);
        
        myPerceptron.getLearningRule().addListener(this);
        System.out.println("Training neural network BinOutputBPNN...");
        myPerceptron.learn(trainingSet);
        
        System.out.println("Testing trained neural network BinOutputBPNN");
        testNeuralNetwork(myPerceptron, trainingSet);
	}
	
	public void testMaxLottery(){
		DataSet trainingSet = new DataSet(49,49);
		DataSet testingSet = new DataSet(49,49);
		double[] dataRow_testIn= new double[49];
		double[] dataRow_testOut= new double[49];
		double maxError = 0.0001d;
		
		ArrayList<String> dataRows = (ArrayList<String>)DataProcessBean.loadMaxBinaryData();
		
		for (int j = 0; j < dataRows.size()-150-1; j++ ){
			double[] dataRow_in= new double[49];
			double[] dataRow_out= new double[49];
			String[] rowSplitIn = dataRows.get(j).split(",");
			String[] rowSplitOut = dataRows.get(j+1).split(",");
			
			for (int i = 0 ; i < 49 ; i++){
				dataRow_in[i] = Double.parseDouble(rowSplitIn[i]);
				dataRow_out[i] = Double.parseDouble(rowSplitOut[i]);
				dataRow_testOut[i] = Double.NaN;
			}
			trainingSet.addRow(new DataSetRow(dataRow_in,dataRow_out));
			
			dataRow_testIn = dataRow_out;
		}
		
		/*
		trainingSet.addRow(new DataSetRow(new double[]{0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0}, 
										  new double[]{0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0}));
        trainingSet.addRow(new DataSetRow(new double[]{0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
        								  new double[]{0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0}));
        trainingSet.addRow(new DataSetRow(new double[]{0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0}, 
        								  new double[]{0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0}));
        trainingSet.addRow(new DataSetRow(new double[]{0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0}, 
        								  new double[]{0,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0}));
        
        trainingSet.addRow(new DataSetRow(new double[]{0,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
        								  new double[]{0,0,0,0,0,1,0,0,1,1,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0}));
        trainingSet.addRow(new DataSetRow(new double[]{0,0,0,0,0,1,0,0,1,1,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0}, 
        								  new double[]{0,1,0,0,0,0,1,1,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}));
        trainingSet.addRow(new DataSetRow(new double[]{0,1,0,0,0,0,1,1,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
        								  new double[]{0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0}));
        trainingSet.addRow(new DataSetRow(new double[]{0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0}, 
        								  new double[]{0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0}));
        */
		
        BinOutputBPNN myPerceptron = new BinOutputBPNN(TransferFunctionType.SIGMOID,49,99,99,99,99,99,99,99,99,99,49);
        myPerceptron.getLearningRule().setMaxError(maxError);
        myPerceptron.getLearningRule().addListener(this);
        System.out.println("Training neural network BinOutputBPNN...");
        myPerceptron.learn(trainingSet);
        
        System.out.println("Testing trained neural network BinOutputBPNN");
        /*
        testingSet.addRow(new DataSetRow(new double[]{0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0}, 
        								  new double[]{Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,
									        		Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,
									        		Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,
									        		Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,
									        		Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,
									        		Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,
									        		Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN}));
		*/
        testingSet.addRow(new DataSetRow(dataRow_testIn,dataRow_testOut));
        testNeuralNetwork(myPerceptron, testingSet);
	}
	
	public void testMaxLottery2(){
		int T = 5; // take T times result to add, calculate the probability
		
		DataSet trainingSet = new DataSet(49,49);
		DataSet testingSet = new DataSet(49,49);
		double[] dataRow_testIn= new double[49];
		double[] dataRow_testOut= new double[49];
		double maxError = 0.0001d;
		double learningRate = 0.6d;
		
		ArrayList<String> dataRows = (ArrayList<String>)DataProcessBean.loadMaxBinaryData();
		
		for (int j = 0; j < dataRows.size()-T-150-1; j++ ){
			double[] dataRow_in= new double[49];
			double[] dataRow_out= new double[49];
			String[] rowSplitOut = dataRows.get(j+T).split(",");
			System.out.println("train out row " + j + ":" + dataRows.get(j+T));
			
			for (int i = 0 ; i < 49 ; i++){
				dataRow_in[i] = 0.0d;
				for(int t = 0; t < T; t++){
					dataRow_in[i] = dataRow_in[i] + Double.parseDouble(dataRows.get(j+t).split(",")[i]);
				}
				dataRow_out[i] = Double.parseDouble(rowSplitOut[i]);
			}
			trainingSet.addRow(new DataSetRow(dataRow_in,dataRow_out));
		}
		
		for (int j = dataRows.size()-T-150; j < dataRows.size()-150; j++ ){
			double[] dataRow_in= new double[49];
			for (int i = 0 ; i < 49 ; i++){
				dataRow_in[i] = 0.0d;
				for(int t = 0; t < T; t++){
					dataRow_in[i] = dataRow_in[i] + Double.parseDouble(dataRows.get(j+t).split(",")[i]);
					System.out.println("Test row " + t + ":" + dataRows.get(j+t));
				}
				dataRow_testOut[i] = Double.NaN;
			}
			dataRow_testIn = dataRow_in;
		}
		
        BinOutputBPNN myPerceptron = new BinOutputBPNN(TransferFunctionType.SIGMOID,49,99,49);
        myPerceptron.getLearningRule().setMaxError(maxError);
        myPerceptron.getLearningRule().setLearningRate(learningRate);
        myPerceptron.getLearningRule().addListener(this);
        System.out.println("Training neural network BinOutputBPNN...");
        myPerceptron.learn(trainingSet);
        
        System.out.println("Testing trained neural network BinOutputBPNN");
        testingSet.addRow(new DataSetRow(dataRow_testIn,dataRow_testOut));
        testNeuralNetwork(myPerceptron, testingSet);
	}
	
	public static void testNeuralNetwork(NeuralNetwork neuralNet, DataSet testSet) {

        for(DataSetRow testSetRow : testSet.getRows()) {
            neuralNet.setInput(testSetRow.getInput());
            neuralNet.calculate();
            double[] networkOutput = neuralNet.getOutput();

            System.out.print("Input: " + Arrays.toString( testSetRow.getInput() ) );
            System.out.println("Output: " + Arrays.toString( networkOutput) );
        }
    }
	
	@Override
	public void handleLearningEvent(LearningEvent event) {
		IterativeLearning bp = (IterativeLearning)event.getSource();
        System.out.println("iterate:"+bp.getCurrentIteration()); 
        System.out.println(Arrays.toString(bp.getNeuralNetwork().getWeights()));
		
	}

}
