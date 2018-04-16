//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Christian McDaniel and Jeremy Shi
 *  @version 1.0
 *  @date    Apr 11, 2018
 *  @see     LICENSE (MIT style license file)
 */

import scala.collection.immutable.ListMap
import scala.math

import scala.util.Random

import scalation.linalgebra.{MatrixD, VectoD, VectorD, VectoI, VectorI}
import scalation.random.RandomVecI
import scalation.linalgebra._
import scalation.plot.Plot
import scalation.random.CDF.studentTCDF
import scalation.util.{banner, Error, time}
import scalation.util.Unicode.sub
import scalation.stat.StatVector.corr
import scalation.util.getFromURL_File
import scalation.util.Error
import scalation.analytics.Regression
import scalation.analytics.SimpleRegression
import scalation.analytics.ExpRegression
import scalation.analytics.LassoRegression
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The 'RegressionTestFinal' object uses the MatrixD and Regression classes
* to perform multiple regression and subsequent analysis on a numerical dataset, read in as an
* argument at the time of running the program.
*  > sbt "run "
*  I hardcode the file path but you can easily change the path
*/
/** WARING **/
/** You must use the 1.4 version of Scalation (released on early March), instead
* of the version 1.4.1, which does not work here!
* If you need the jar file, let me know.
*/


object RegressionTestWhite extends App
{

    val data_path = "/Users/yuanmingshi/Downloads/winequality-white.csv" // take in filepath as argument

    // set up your training and test datasets and variables
    val df = MatrixD(data_path, 1)   //skip the first row!

    // Method to find x and y for regression
    def buildRegression (df: MatrixD): (MatrixD, VectoD) =
    {
      val rows = df.dim1  // number of rows
      val columns = df.dim2 //number of columns
      val df_x = df.sliceExclude(rows, columns-1)  // delete the last column (because it's Y)
      val df_y = df.col(columns-1) //select the last column of original df as Y
      val df_i = df_y.copy  // copy the Y vector into another vector (for calculating the intercept)
      df_i.set(1) // set all values in the the new copied vector into 1
      val df_x1 = df_x.+^:(df_i) //concat the x matrix with the new vector
      (df_x1, df_y)
    }

    println("**************************")
    println("*   Linear Regression    *")
    println("**************************")

    val (df_x1, df_y) = buildRegression(df)
    val rg = new Regression(df_x1, df_y) //set a new regression
    rg.train()
    rg.report()
    println ("vif values:")
    // println
    println (rg.vif)
    // get the new var with cut
    var df_new = df_x1
    println (df_new.dim1, df_new.dim2)
    for (i <- 0 until df_x1.dim2 - 1) {
      if (rg.vif(i) >= 10){
        df_new = df_new.sliceExclude(df_x1.dim1, i)
      }
    }

    // perform an updated regression on the revised dataset
    val new_rg = new Regression(df_new, df_y) //set a new regression
    println ("After removing features with vif values over 10, the results are:")
    new_rg.train()
    new_rg.report()
    println ("vif values:")
    println (new_rg.vif)


    // Exponential Regression
    println("**************************")
    println("* Exponential Regression *")
    println("**************************")
    // Apparently, Exponential Regression doesn't work well
    // using Exponential Regression
    val ex_rg = new ExpRegression(df_x1, true, df_y)
    ex_rg.train()
    print (ex_rg.metrics)
}


object RegressionTestRed extends App
{

    val data_path = "/Users/yuanmingshi/Downloads/winequality-red.csv" // take in filepath as argument

    val df = MatrixD(data_path, 1)   //skip the first row!

    def buildRegression (df: MatrixD): (MatrixD, VectoD) =
    {
      val rows = df.dim1  // number of rows
      val columns = df.dim2 //number of columns
      val df_x = df.sliceExclude(rows, columns-1)  // delete the last column (because it's Y)
      val df_y = df.col(columns-1) //select the last column of original df as Y
      val df_i = df_y.copy  // copy the Y vector into another vector (for calculating the intercept)
      df_i.set(1) // set all values in the the new copied vector into 1
      val df_x1 = df_x.+^:(df_i) //concat the x matrix with the new vector
      (df_x1, df_y)
    }

    println("**************************")
    println("*   Linear Regression    *")
    println("**************************")

    val (df_x1, df_y) = buildRegression(df)
    // // run your initial regression
    val rg = new Regression(df_x1, df_y) //set a new regression
    rg.train()
    rg.report()
    println ("vif values:")
    // println
    println (rg.vif)
    var df_new = df_x1
    println (df_new.dim1, df_new.dim2)
    for (i <- 0 until df_x1.dim2 - 1) {
      if (rg.vif(i) >= 10){
        df_new = df_new.sliceExclude(df_x1.dim1, i)
      }
    }

    // perform an updated regression on the revised dataset
    val new_rg = new Regression(df_new, df_y) //set a new regression
    println ("After removing features with vif values over 10, the results are:")
    new_rg.train()
    new_rg.report()
    println ("vif values:")
    println (new_rg.vif)


    // Exponential Regression
    println("**************************")
    println("* Exponential Regression *")
    println("**************************")
    // Apparently, Exponential Regression doesn't work well
    // using Exponential Regression
    val ex_rg = new ExpRegression(df_x1, true, df_y)
    ex_rg.train()
    print (ex_rg.metrics)
}
