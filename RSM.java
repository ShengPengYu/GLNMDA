package cn.rocket.glnp;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import cn.rocket.glnp.utils.GUIutils;

/**
 * function:Rebuild the Similarity Matrix by GLNP
 * 
 * @author rocket
 *
 */
public class RSM {
	private static final int D_K = 52;
	private static final int M_K = 52;

	public static final INDArray S_D = getRebuildDSimilMatrix(Datas.INT.dup());
	// public static final INDArray S_D = RSM_D(Datas.INT.dup()) ;
	public static final INDArray S_M = getRebuildMSimilMatrix(Datas.INT.transpose().dup());

	public static INDArray getRebuildDSimilMatrix(INDArray matrix) {
		INDArray last = Nd4j.ones(matrix.rows(), D_K);
		INDArray next = null;
		double error = 1;
		while (error > Parameters.EPSILON) { // Iterating until convergence
			next = last
					.mul(Transforms.pow(
							matrix.mmul(matrix.transpose()).mmul(last).mul(2).div(last.mmul(last.transpose())
									.mmul(matrix).mmul(matrix.transpose()).mmul(last).add(matrix
											.mmul(matrix.transpose()).mmul(last).mmul(last.transpose()).mmul(last))),
							1.0 / 2));
			error = Transforms.pow(last.sub(next), 2.0).sumNumber().doubleValue(); // Calculate the error between the
																					// last result and the next result
			last = next;
			System.out.println(error);
		}

		INDArray result = last.mmul(last.transpose());
		result = Datas.normalizeMatrix(result, Datas.getDegreeMatrix(result));

		for (int i = 0; i < result.rows(); i++) {
			for (int j = 0; j < result.columns(); j++) {
				if (Datas.DSS.getDouble(i, j) != 0) {
					 result.putScalar(i, j,(Datas.S_D.getDouble(i,j)+result.getDouble(i,j)/2.0)) ;
					 //result.putScalar(i, j,Datas.DSS.getDouble(i,j)) ;
					//result.putScalar(i, j, Datas.S_D.getDouble(i, j));
				}
			}
		}
		
		result = Datas.normalizeMatrix(result, Datas.getDegreeMatrix(result));
		return result;
	}

	public static INDArray getRebuildMSimilMatrix(INDArray matrix) {
		INDArray last = Nd4j.ones(matrix.rows(), M_K);
		INDArray next = null;
		double error = 1;
		while (error > Parameters.EPSILON) { // Iterating until convergence
			next = last
					.mul(Transforms.pow(
							matrix.mmul(matrix.transpose()).mmul(last).mul(2).div(last.mmul(last.transpose())
									.mmul(matrix).mmul(matrix.transpose()).mmul(last).add(matrix
											.mmul(matrix.transpose()).mmul(last).mmul(last.transpose()).mmul(last))),
							1.0 / 2));
			error = Transforms.pow(last.sub(next), 2.0).sumNumber().doubleValue(); // Calculate the error between the
																					// last result and the next result

			last = next;
			System.out.println(error);
		}

		INDArray result = last.mmul(last.transpose());
		result = Datas.normalizeMatrix(result, Datas.getDegreeMatrix(result));

		for (int i = 0; i < result.rows(); i++) {
			for (int j = 0; j < result.columns(); j++) {
				if (Datas.MFS.getDouble(i, j) != 0) {
					 result.putScalar(i, j,(Datas.S_M.getDouble(i,j)+result.getDouble(i,j)/2.0)) ;
					 //result.putScalar(i,j,Datas.MFS.getDouble(i,j)) ;
				}
			}
		}
		
		result = Datas.normalizeMatrix(result, Datas.getDegreeMatrix(result));
		
		return result;
	}

	public static void main(String[] args) {
		GUIutils.showMatrixGUI(RSM.S_D, "Predict result");
	}
}
