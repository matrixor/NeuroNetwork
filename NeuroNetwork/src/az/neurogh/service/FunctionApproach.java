package az.neurogh.service;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.core.learning.SupervisedLearning;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.FileUtils;
import org.neuroph.util.TransferFunctionType;

import az.neurogh.demo.BPNeuroNetwork;
import az.neurogh.demo.LinOutputBPNN;

public class FunctionApproach implements LearningEventListener{
	BPNeuroNetwork myMlPerceptron;
	public static double maxError=0.0001d;
	public static int i=2;
	//public static int i=4;
	
	public FunctionApproach(){
		init();
	}
	
	private void init(){
		this.myMlPerceptron = new LinOutputBPNN(TransferFunctionType.SIGMOID, 1, 4 , 1);
		//this.myMlPerceptron = new LinOutputBPNN(TransferFunctionType.SIGMOID, 1, 8 , 1); // when i = 4
		myMlPerceptron.setLearningRule(new BackPropagation()); 
		LearningRule learningRule = myMlPerceptron.getLearningRule();
        learningRule.addListener(this);
        ((SupervisedLearning)learningRule).setMaxError(maxError);
	}
	
	public void train() { 
		DataSet trainingSet = new DataSet(1, 1);
        for(int i=0;i<2000;i++){
        	double in=new Random().nextDouble()*4-2;
        	double out=targetFunc(in);
        	trainingSet.addRow(new DataSetRow(new double[]{in}, new double[]{out}));
        }

        System.out.println("Training neural network...");
        myMlPerceptron.learn(trainingSet);
	}
	
	public void test() throws IOException {
		System.out.println("Testing trained neural network");
		
		StringBuffer x=new StringBuffer();
    	StringBuffer y=new StringBuffer();
    	for(int i=0;i<100;i++){
    		double in=new Random().nextDouble()*4-2;
    		myMlPerceptron.setInput(in);
    		myMlPerceptron.calculate();
    	    double[] networkOutput = myMlPerceptron.getOutput();
    	    
    	    x.append(in);
    	    x.append("\t");
    	    y.append(networkOutput[0]);
    	    y.append("\t");
    	}
    	
    	FileUtils.writeStringToFile(new File("/x.txt"), x.toString());
    	FileUtils.writeStringToFile(new File("/y.txt"), y.toString());
	}
	
	public double targetFunc(double x){
		return 1+Math.sin(Math.PI*i/4*x);
	}

	@Override
	public void handleLearningEvent(LearningEvent event) {
		SupervisedLearning bp = (SupervisedLearning)event.getSource();
        System.out.println(bp.getCurrentIteration() + ". iteration : "+ bp.getTotalNetworkError());	
	}

}
