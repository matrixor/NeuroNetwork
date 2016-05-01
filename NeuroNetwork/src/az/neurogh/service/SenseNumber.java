package az.neurogh.service;

import java.util.Arrays;
import java.util.Random;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.core.learning.SupervisedLearning;
//import org.neuroph.core.learning.IterativeLearning;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

import az.neurogh.demo.BPNeuroNetwork;
import az.neurogh.util.FunctionTools;

public class SenseNumber implements LearningEventListener {
	BPNeuroNetwork myMlPerceptron;
	
	public SenseNumber(){
		init();
	}
	
	private void init(){
		this.myMlPerceptron = new BPNeuroNetwork(TransferFunctionType.SIGMOID, 32, 10, 4);
		LearningRule learningRule = myMlPerceptron.getLearningRule();
        learningRule.addListener(this);
        ((SupervisedLearning)learningRule).setMaxError(0.0001d);
	}
	
	public void train() { 
        DataSet trainingSet = new DataSet(32, 4);
        
        for(int i=0;i<2000;i++){
        	int in=new Random().nextInt();
        	trainingSet.addRow(new DataSetRow(FunctionTools.int2double(in), FunctionTools.int2prop(in)));
        }
        
        System.out.println("Training neural network...");
        myMlPerceptron.learn(trainingSet);
	}
	
	public void test() {
		System.out.println("Testing trained neural network");
		int badcount=0;
    	int COUNT=5000;
    	
    	for(int i=0;i<COUNT;i++){
    		int in=new Random().nextInt();
    		double[] inputnumber=FunctionTools.int2double(in);
    		myMlPerceptron.setInput(inputnumber);
    		myMlPerceptron.calculate();
    	    double[] networkOutput = myMlPerceptron.getOutput();
    	    
    	    networkOutput=FunctionTools.competition(networkOutput);
    	    
    	    String networkOutputDisplay=networkOutputDisplay(networkOutput);
    	    String cc=FunctionTools.classifyInt(in);
    	    
    	    System.out.print(in+" "+networkOutputDisplay+" ");
    	    if(i%50==0){
    	    	System.out.println();
    	    }
    	    
    	    if(!cc.equals(networkOutputDisplay)){
    	    	badcount++;
    	    	System.out.print("classify error:"+in);
    	    	System.out.print(" correctClassify="+cc);
    	    	System.out.println(" networkOutputDisplay="+networkOutputDisplay);
    	    }
    	    
    	    System.out.println();
        	System.out.println("Correct Rate: "+(COUNT-badcount*1.0)/COUNT*100.0+"%");    
    	}
	}
	
	public void run(int in) {
		double[] inputnumber=FunctionTools.int2double(in);
		myMlPerceptron.setInput(inputnumber);
		myMlPerceptron.calculate();
	    double[] networkOutput = myMlPerceptron.getOutput();
	    networkOutput=FunctionTools.competition(networkOutput);
	    String networkOutputDisplay=networkOutputDisplay(networkOutput);
	    
	    System.out.println(in+" "+networkOutputDisplay+" ");
	}
	
	public static String networkOutputDisplay(double[] networkOutput){
		if(((int)networkOutput[3])==1)return "positive even number";
		if(((int)networkOutput[2])==1)return "negative even number";
		if(((int)networkOutput[1])==1)return "positive odd number";
		if(((int)networkOutput[0])==1)return "negative odd number";
		return "unknow number";
	}
	
	@Override
	public void handleLearningEvent(LearningEvent event) {
		BackPropagation bp = (BackPropagation)event.getSource();
		//IterativeLearning bp = (IterativeLearning)event.getSource();
        System.out.println(bp.getCurrentIteration() + ". iteration : "+ bp.getTotalNetworkError()); 
        System.out.println(Arrays.toString(bp.getNeuralNetwork().getWeights()));
	}

}
