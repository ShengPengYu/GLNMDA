package cn.rocket.glnp.utils;

import java.io.File;
import java.io.IOException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;
import org.ujmp.jmatio.ImportMatrixMAT;

/**
 * 
 * @author ShengPeng Yu
 * @data: 2017/12/1
 * 
 */
public class LoadDataUtils {
	public static INDArray mat2Ndarray(String filename) {
		INDArray result = null ;		
		File file = new File(filename);
		Matrix dssmatrix;
		try {
			dssmatrix = ImportMatrixMAT.fromFile(file);
			result =Nd4j.create(dssmatrix.toDoubleArray());
		} catch (IOException e) {
			
			e.printStackTrace();
			System.out.println("加载数据异常，SystemUtis:mat2Ndarray");
		}
		
		return result ;
	}
	
	public static INDArray[] svd(INDArray matrix){
		INDArray[] arrays = null ;
		
		double[][] data = new double[matrix.rows()][matrix.columns()];
		for(int i = 0 ; i < matrix.rows() ; i++) {
			for(int j = 0 ; j < matrix.columns() ;j++)
				data[i][j] = matrix.getDouble(i, j);
		}
		
		Matrix svd_matrix = Matrix.Factory.importFromArray(data);
		
		Matrix[] svd_result = svd_matrix.svd();
		arrays = new INDArray[svd_result.length] ;
		
		for(int k = 0 ; k < arrays.length ; k++) {
			arrays[k] = Nd4j.create(svd_result[k].toDoubleArray()) ;
		}
		
		return arrays ;
	}
	
	public static INDArray[] eig(INDArray matrix){
		INDArray[] arrays = null ;
		
		double[][] data = new double[matrix.rows()][matrix.columns()];
		for(int i = 0 ; i < matrix.rows() ; i++) {
			for(int j = 0 ; j < matrix.columns() ;j++)
				data[i][j] = matrix.getDouble(i, j);
		}
		
		Matrix svd_matrix = Matrix.Factory.importFromArray(data);
		
		Matrix[] svd_result = svd_matrix.eig();
		arrays = new INDArray[svd_result.length] ;
		
		for(int k = 0 ; k < arrays.length ; k++) {
			arrays[k] = Nd4j.create(svd_result[k].toDoubleArray()) ;
		}
		
		return arrays ;
	}
	
	public static INDArray cosSimilarity(INDArray matrix) {
		double[][] data = new double[matrix.rows()][matrix.columns()];
		
		for(int i = 0 ; i < matrix.rows() ; i++) {
			for(int j = 0 ; j < matrix.columns() ;j++)
				data[i][j] = matrix.getDouble(i, j);
		}
		Matrix m = Matrix.Factory.importFromArray(data);
		return Nd4j.create(m.cosineSimilarity(Ret.LINK, true).toDoubleArray());
	}
}
