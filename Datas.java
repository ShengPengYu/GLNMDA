package cn.rocket.glnp;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import cn.rocket.glnp.utils.GUIutils;
import cn.rocket.glnp.utils.LoadDataUtils;

/*

 */
/**
 * function:This Class be used to storage all datas used in experiment
 * @author YuShengPeng
 *
 */
public class Datas {
	public static final INDArray DSS = LoadDataUtils.mat2Ndarray("resources/dss.mat");              //All disease	
	public static final INDArray MFS = LoadDataUtils.mat2Ndarray("resources/mfs.mat");              //All RNA
	public static final INDArray INT = LoadDataUtils.mat2Ndarray("resources/interaction.mat") ;     //All interactions
	
	
	public static final Integer D_SIZE = DSS.columns() ;                                            //Size of disease
	public static final Integer M_SIZE = MFS.columns() ;                                            //Size of RNA
	
	public static final ArrayList<String> DSS_NAME = gettNames("resources/dssname.txt");            //All Disease names
	public static final ArrayList<String> MFS_NAME = gettNames("resources/mfsname.txt");            //All RNA names
	
	public static final INDArray ADJ_D = getAdjMatrix(DSS).add(Nd4j.eye(D_SIZE)) ;                  //Adjacent matrix of Diseases 
	public static final INDArray ADJ_M = getAdjMatrix(MFS).add(Nd4j.eye(M_SIZE)) ;					//Adjacent matrix of RNAs

	public static final INDArray D_D = getDegreeMatrix(DSS)  ;                                      //Degree matrix of dssMatrix
	public static final INDArray D_M = getDegreeMatrix(MFS) ;                                       //Degree matrix of mfsMatrix
	
	public static final INDArray S_D = normalizeMatrix(DSS,D_D) ;                                   //Similarity matrix of Diseases after normalization
	public static final INDArray S_M =normalizeMatrix(MFS,D_M) ;									//Similarity matrix of RNAs after normalization
	
	public static final INDArray L_D = getLaplacianMatrix(D_D, ADJ_D) ;                             //Laplacian matrix of dssMatrix
	public static final INDArray L_M = getLaplacianMatrix(D_M, ADJ_M) ;                             //Laplacian matrix of mfsMatrix
	public static final Integer KNOWN_EDGE_SIZE = getKnownEdgeSize() ;
	public static final INDArray KNOWN_EDGE = getKnownEdge();
	public static INDArray normalizeMatrix(INDArray matrix,INDArray degreeMatrix) {
		INDArray result =Nd4j.zeros(matrix.rows(), matrix.columns()) ;
		for(int i = 0 ; i < matrix.rows(); i++) result.putScalar(i, i, Math.pow(degreeMatrix.getDouble(i, i), -1.0/2));
		result =result.mmul(matrix).mmul(result);
		return result ;
	}
	
	public static INDArray getDegreeMatrix(INDArray matrix) {
		INDArray result = Nd4j.zeros(matrix.rows(),matrix.columns());
		for(int i = 0 ; i < matrix.rows() ;i++) {
			result.putScalar(i, i, matrix.getRow(i).sumNumber().doubleValue());
		}
		return result;
	}

	private static INDArray getAdjMatrix(INDArray matrix) {
		INDArray result = Nd4j.zeros(matrix.rows(),matrix.columns());
		for(int i = 0 ; i < matrix.rows() ;i++) 	
			for(int j = 0 ; j < matrix.columns() ;j++) 
				if(matrix.getDouble(i,j)>0)
					result.putScalar(i, j,1);
		return result;
	}
	
	
	private static INDArray getLaplacianMatrix(INDArray degreeMatrix, INDArray saimilMatrix) {
		INDArray matrix = degreeMatrix.dup();
		for(int i = 0 ; i < matrix.rows() ;i++) 
				matrix.putScalar(i, i,1/Math.sqrt(matrix.getDouble(i,i)));
		
		INDArray result = matrix.mmul(degreeMatrix.sub(saimilMatrix)).mmul(matrix);
		return result ;
	}
	
	public static Integer getKnownEdgeSize() {
		int count = 0 ;
		for(int i = 0 ; i < Datas.INT.rows() ; i++){
			for(int j = 0 ; j < Datas.INT.columns() ; j++){
				if(Datas.INT.getDouble(i,j)==1.0){
					count++ ;
				}
			}
		}
		return count ;
	}
	
	public static INDArray getKnownEdge() {
		int count = 0 ;
		INDArray knowEdge = Nd4j.zeros(Datas.KNOWN_EDGE_SIZE, 2);
		for(int i = 0 ; i < Datas.INT.rows() ; i++){
			for(int j = 0 ; j < Datas.INT.columns() ; j++){
				if(Datas.INT.getDouble(i,j)==1.0){
					knowEdge.putScalar(count, 0, i);
					knowEdge.putScalar(count, 1, j);
					count ++ ;
				}
			}
		}
		return knowEdge;
	}
	
	
	public static ArrayList<String> gettNames(String filename)  {
		ArrayList<String> result = new ArrayList<>();
		try {
			InputStreamReader read = new InputStreamReader(new FileInputStream(filename));
	        
			BufferedReader input = new BufferedReader(read);
	        String line = null ;
			while((line = input.readLine())!=null)
				result.add(line.trim());
			read.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return result ;
	}
	
	public static void main(String[] args) throws IOException {
		System.out.println(Arrays.toString(Datas.MFS.shape()));
		System.out.println(Datas.MFS_NAME.size());
	}
}
