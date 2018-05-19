package cn.rocket.glnp.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.ujmp.core.Matrix;

public class GUIutils {
	//用JFrame显示数组信息
	public static  void showMatrixGUI(INDArray matrix,String title) {
		double[][] data = new double[matrix.rows()][matrix.columns()];
		
		for(int i = 0 ; i < matrix.rows() ; i++) {
			for(int j = 0 ; j < matrix.columns() ;j++)
				data[i][j] = matrix.getDouble(i, j);
		}
		
		Matrix gui_matrix = Matrix.Factory.importFromArray(data);
		gui_matrix.getGUIObject().setLabel(title);
		gui_matrix.showGUI() ;
		
	}
	
	public static void splitLine(char c,Integer length) {
		StringBuffer stringBuffer = new StringBuffer() ;
		for(int i = 0 ; i < length ; i++) {
			stringBuffer.append(c);
		}
		System.out.println(stringBuffer);
	}
	public static void splitLine(char c) {
		StringBuffer stringBuffer = new StringBuffer() ;
		for(int i = 0 ; i < 50 ; i++) {
			stringBuffer.append(c);
		}
		System.out.println(stringBuffer);
	}
	
	
}




