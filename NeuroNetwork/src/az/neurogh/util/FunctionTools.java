package az.neurogh.util;

public class FunctionTools {
	
	// convert int to 32bit binary
	public static double[] int2double(int i){
		double[] re=new double[32];
		for(int j=0;j<32;j++){
			re[j]=(double)((i>>j)&1);
		}
		return re;
	}
	
	/**
	 * 0001 positive even number
	 * 0010 negative even number
	 * 0100 positive odd number
	 * 1000 negative odd number
	 * @param i
	 * @return
	 */
	public static double[] int2prop(int i){
		double[] pe={0d,0d,0d,1d};
		double[] ne={0d,0d,1d,0d};
		double[] po={0d,1d,0d,0d};
		double[] no={1d,0d,0d,0d};
		if(i>0 && i%2==0){
			return pe;
		}else if(i<0 && i%2==0){
			return ne;
		}else if(i>0 && i%2!=0){
			return po;
		}else if(i<0 && i%2!=0){
			return no;
		}
		return pe;
	}
	
	public static String classifyInt(int i){
		if(i>0 && i%2==0){
			return "positive even number";
		}else if(i<0 && i%2==0){
			return "negative even number";
		}else if(i>0 && i%2!=0){
			return "positive odd number";
		}else if(i<0 && i%2!=0){
			return "negative odd number";
		}
		return "0";
	}
	
	/**
	 * convert the max in the vector to 1, others to 0
	 * @param d
	 * @return
	 */
	public static double[] competition(double[] d){
		double[] output=d;
		double[] re=new double[output.length];
		int maxIndex=0;
		double maxValue=Double.MIN_VALUE;
		for(int i=0;i<output.length;i++){
			if(output[i] > maxValue){
				maxIndex=i;
				maxValue=output[i];
			}
		}
		for(int i=0;i<re.length;i++){
			if(i==maxIndex){
				re[i]=1;
			}else{
				re[i]=0;
			}
		}
        return re;
	}
}
