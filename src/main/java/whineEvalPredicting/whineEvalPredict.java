package whineEvalPredicting;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class whineEvalPredict {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		
		String[] classNames = {"10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"};
		
		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("whineQuality.zip"));
		
		INDArray predictData = Nd4j.create(new double[][]{
			{2,0.33,0.12,10.6,0.045,12,90,0.98345,1.99,0.27,9.4},
			{4,0.75,0.20,11.7,0.55,44,124,0.99732,0.54,0.41,11.8},
			{6,0.02,0.91,13.9,0.002,63,154,0.96543,3.67,0.6,8.2}
		});
		
		INDArray output = model.output(predictData);
		int[] classes = output.argMax(1).toIntVector();
		System.out.println(output);
		for(int i=0; i<classes.length; i++) {
			System.out.println("Quality of Whine is : "+ classNames[classes[i]]);
		}
	}
}



