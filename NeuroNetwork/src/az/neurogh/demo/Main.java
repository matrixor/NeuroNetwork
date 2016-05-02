package az.neurogh.demo;

import java.io.IOException;

import az.neurogh.service.FunctionApproach;
import az.neurogh.service.SenseNumber;

public class Main {

	public static void main(String[] args) {
		
		//TestBean testBean = new TestBean();
		//testBean.testSimplePerceptron_And();
		//testBean.testSimplePerceptron_Or();
		//testBean.testMultiPerceptron_AndOutputNoLearn();
		//testBean.testBinOutputBPNN();
		
		/*
		SenseNumber senseNumber = new SenseNumber();
		senseNumber.train();
		senseNumber.test();
		senseNumber.run(1234);
		senseNumber.run(1233);
		senseNumber.run(-987654);
		senseNumber.run(-987653);
		senseNumber.run(-11111111);
		senseNumber.run(-11111112);
		senseNumber.run(-11111113);
		*/
		
		FunctionApproach functionApproach = new FunctionApproach();
		functionApproach.train();
		
		try {
			functionApproach.test();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

}
