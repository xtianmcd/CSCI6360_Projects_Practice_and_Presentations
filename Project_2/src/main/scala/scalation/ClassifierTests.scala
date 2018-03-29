
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Christian McDaniel
 *  @version 1.4
 *  @date    14 Mar 2018
 *  @see     LICENSE (MIT style license file).
 */

package scalation

import scalation.analytics.classifier.{NullModel, NaiveBayes, NaiveBayes0, TANBayes, TANBayes0, ClassifierInt, LDA, LogisticRegression, KNN_Classifier}
import scalation.columnar_db.Relation
import scalation.random.PermutedVecI
import scalation.random.RNGStream.ranStream
import scalation.linalgebra._
import scala.math.sqrt

object ClassifierTests extends App
{
    private val BASE_DIR = "../data/analytics/classifier/"

    // load the ExampleTennis dataset
    import scalation.analytics.classifier.ExampleTennis._
    val xyTennis    = xy
    val xTennis     = xyTennis                  (0 until xyTennis.dim1, 0 until xyTennis.dim2 - 1)
    val yTennis     = xyTennis.col              (xyTennis.dim2 - 1)
    val fnTennis    = fn
    val cnTennis    = cn

    // load and prepare the Caravan dataset
    val caravanFP   = BASE_DIR                  + "caravan.csv"
    val dataCaravan = Relation                  (caravanFP, "Caravan", -1, null, ",")
    val xyCaravan   = dataCaravan.toMatriI2     (null)
    val yCaravan    = xyCaravan.col             (xyCaravan.dim2 - 1)
    val fnCaravan   = dataCaravan.colName.slice (0, xyCaravan.dim2-1).toArray
    val cnCaravan   = Array                     ("No", "Yes")
    val xCaravan    = xyCaravan                 (0 until xyCaravan.dim1, 0 until xyCaravan.dim2 - 1)

    // load and prepare the breast-cancer.arff data
    val brstCncrDta = BASE_DIR                  + "breast-cancer.arff"
    var dataCancer  = Relation                  (brstCncrDta, -1, null)
    val xyCancer    = dataCancer.toMatriI2      (null)
    val xCancer     = xyCancer                  (0 until xyCancer.dim1, 0 until xyCancer.dim2 - 1)
    val yCancer     = xyCancer.col              (xyCancer.dim2 - 1)
    val fnCancer    = dataCancer.colName.slice  (0, xyCancer.dim2 - 1).toArray
    val cnCancer    = Array                     ("p", "e")                        // class names

    val k = 2

    def getStats (tp :Double, tn :Double, fp :Double, fn :Double): (Double, Double, Double, Double) =
    {
      var foldAcc    = (tp + tn) / (tp + tn + fp + fn)
      var foldRecall = 0.0
      var foldPrecis = 0.0
      var foldFScore = 0.0

      if (tp == 0.0) {
          if ((fp == 0.0) & (fn == 0.0)) {
              foldRecall = 1.0
              foldPrecis = 1.0
              foldFScore = 1.0
          } else {
              foldRecall = 0.0
              foldPrecis = 0.0
              foldFScore = 0.0
          }
      } else {
        foldRecall = tp / (tp + fn)
        foldPrecis = tp / (tp + fp)
        foldFScore = 2 * foldPrecis * foldRecall / (foldPrecis + foldRecall)
      } // else tp not 0

      (foldAcc, foldRecall, foldPrecis, foldFScore)
    } // getStats

    def checkClass (predictArray: VectorD, actualArray: VectorD, show: Boolean = false): (Double, Double, Double, Double) =
      {
        if (show) {
          println(s"predictions: $predictArray")
          println(s"class: $actualArray")
        } // if show

        var fp = 0.0
        var fn = 0.0
        var tp = 0.0
        var tn = 0.0

        for (pred <- 0 until predictArray.dim)  {

          if (predictArray(pred) == 0.0) {
              if (actualArray(pred) == 0)      tn += 1.0
              else if (actualArray(pred) == 1) fn += 1.0
          } // if predict neg

          else if (predictArray(pred) == 1.0) {
              if (actualArray(pred) == 1) tp += 1.0
              else if (actualArray(pred) == 0) fp += 1.0
          } // if predict pos

        } // for

        if (show) {
          println(s"tp: $tp, fp: $fp, tn: $tn, fn: $fn")
        } // if show

        var (acc, recall, precision, fScore) = getStats(tp, tn, fp, fn)
        (acc, recall, precision, fScore)
      } // checkClass

    def crossValidateRand2 (dataset :String, datasetxy :MatriI, datasetx :MatriI, datasety :VectoI, datasetfn :Array[String], datasetcn :Array[String], nx :Int, caravan: Boolean = false, show: Boolean = false)
        {

            val vc          = (for(j <- datasetx.range2) yield datasetx.col(j).max() + 1).toArray
            if (show) {for (i <- vc) println(s"value count at index $i: $vc(i)")}

            val permutedVec = PermutedVecI     (VectorI.range (0, datasetx.dim1), ranStream)
            val randOrder   = permutedVec.igen                       // randomize integers 0 until size
            val itestA      = randOrder.split  (nx)                   // make array of itest indices

            if (show) {for (i <- itestA) println(s"randomly generated list of indices for the fold-wise testing: $i")}

            var accNM       = new VectorD (0)
            var recallNM    = new VectorD (0)
            var precisNM    = new VectorD (0)
            var fScoreNM    = new VectorD (0)

            var accNB       = new VectorD (0)
            var recallNB    = new VectorD (0)
            var precisNB    = new VectorD (0)
            var fScoreNB    = new VectorD (0)

            var accTAN      = new VectorD (0)
            var recallTAN   = new VectorD (0)
            var precisTAN   = new VectorD (0)
            var fScoreTAN   = new VectorD (0)

            var accLDA      = new VectorD (0)
            var recallLDA   = new VectorD (0)
            var precisLDA   = new VectorD (0)
            var fScoreLDA   = new VectorD (0)

            var accLR       = new VectorD (0)
            var recallLR    = new VectorD (0)
            var precisLR    = new VectorD (0)
            var fScoreLR    = new VectorD (0)

            var accKNN      = new VectorD (0)
            var recallKNN   = new VectorD (0)
            var precisKNN   = new VectorD (0)
            var fScoreKNN   = new VectorD (0)

            for (it <- 0 until nx) {                // for loop for cross validation

                var classesNM  = new VectorD (0)
                var classesNB  = new VectorD (0)
                var classesTAN = new VectorD (0)
                var classesLDA = new VectorD (0)
                var classesLR  = new VectorD (0)
                var classesKNN = new VectorD (0)

                val itest      = itestA(it)() 
                if (show) {println(s"randomly generated validation set for current fold: $itest")}                           // get array from it element
                var rowsList   = Array[Int]() 
                for (rowi <- 0 until datasetx.dim1) {
                    if (!itest.contains(rowi)) {rowsList = rowsList :+ rowi}
                } // for    
                if (show) println(s"rows to include for training: length = $rowsList.length")

                val foldx  = datasetx.selectRows(rowsList)
                val foldxR = Relation.fromMatriI (foldx, "foldx", datasetfn)
                val foldxI = foldxR.toMatriD (0 until foldx.dim2)
                val foldxD = new MatrixD (foldxI)
                val foldy  = datasety.select(rowsList)
                if (show) println(s"initial dataset has $datasetx.dim1 rows; fold training set has $foldx.dim1 rows")
                if (show) {println(s"MatrixI training set: $foldx\nMatrixD training set: $foldxD\nTraining labels: $foldy\n")}

                var nm  = new NullModel          (        foldy,            k, datasetcn)
                var nb  = new NaiveBayes         (foldx,  foldy, datasetfn, k, datasetcn,    vc)
                var tan = new TANBayes           (foldx,  foldy, datasetfn, k, datasetcn, 0, vc)
                var lda = new LDA                (foldxD, foldy, datasetfn,    datasetcn)
                var lr  = new LogisticRegression (foldxD, foldy, datasetfn)
                var knn = new KNN_Classifier     (foldxD, foldy, datasetfn, k, datasetcn)

                nm.train  ()
                nb.train  ()
                tan.train ()                                    // train 
                lda.train ()
                if (caravan) {lr.train_SGD ()}
                else {lr.train  ()}
                knn.train ()

                var rowy :Int = 1000
                var yArray    = new VectorD (0)

                //for (ic <- 0 until itest.length) {    // for loop for classifying each example in the validation set
                for (ic <- itest) {

                    var rowx = datasetx (ic)
                    rowy     = datasety (ic)

                    if (show) println(s"row: $rowx")
                    
                    var (iclassNM,  icnNM,  iporbNM)  = nm.classify  ( rowx )
                    var (iclassNB,  icnNB,  iprobNB)  = nb.classify  ( rowx )
                    var (iclassTAN, icnTAN, iprobTAN) = tan.classify ( rowx )
                    var (iclassLDA, icnLDA, iprobLDA) = lda.classify ( rowx )
                    var (iclassLR,  icnLR,  iprobLR)  = lr.classify  ( rowx )
                    var (iclassKNN, icnKNN, iprobKNN) = knn.classify ( rowx )

                    yArray     = yArray     ++ rowy
                    classesNM  = classesNM  ++ iclassNM
                    classesNB  = classesNB  ++ iclassNB
                    classesTAN = classesTAN ++ iclassTAN
                    classesLDA = classesLDA ++ iclassLDA
                    classesLR  = classesLR  ++ iclassLR
                    classesKNN = classesKNN ++ iclassKNN

                } // for classify

                var (foldAccNM,  foldRecallNM,  foldPrecisionNM,  foldFScoreNM)  = checkClass (classesNM,  yArray, show)
                var (foldAccNB,  foldRecallNB,  foldPrecisionNB,  foldFScoreNB)  = checkClass (classesNB,  yArray, show)
                var (foldAccTAN, foldRecallTAN, foldPrecisionTAN, foldFScoreTAN) = checkClass (classesTAN, yArray, show)
                var (foldAccLDA, foldRecallLDA, foldPrecisionLDA, foldFScoreLDA) = checkClass (classesLDA, yArray, show)
                var (foldAccLR,  foldRecallLR,  foldPrecisionLR,  foldFScoreLR)  = checkClass (classesLR,  yArray, show)
                var (foldAccKNN, foldRecallKNN, foldPrecisionKNN, foldFScoreKNN) = checkClass (classesKNN, yArray, show)

                accNM     = accNM     ++ foldAccNM
                recallNM  = recallNM  ++ foldRecallNM
                precisNM  = precisNM  ++ foldPrecisionNM
                fScoreNM  = fScoreNM  ++ foldFScoreNM

                accNB     = accNB     ++ foldAccNB
                recallNB  = recallNB  ++ foldRecallNB
                precisNB  = precisNB  ++ foldPrecisionNB
                fScoreNB  = fScoreNB  ++ foldFScoreNB

                accTAN    = accTAN    ++ foldAccTAN
                recallTAN = recallTAN ++ foldRecallTAN
                precisTAN = precisTAN ++ foldPrecisionTAN
                fScoreTAN = fScoreTAN ++ foldFScoreTAN

                accLDA    = accLDA    ++ foldAccLDA
                recallLDA = recallLDA ++ foldRecallLDA
                precisLDA = precisLDA ++ foldPrecisionLDA
                fScoreLDA = fScoreLDA ++ foldFScoreLDA

                accLR     = accLR     ++ foldAccLR
                recallLR  = recallLR  ++ foldRecallLR
                precisLR  = precisLR  ++ foldPrecisionLR
                fScoreLR  = fScoreLR  ++ foldFScoreLR

                accKNN    = accKNN    ++ foldAccKNN
                recallKNN = recallKNN ++ foldRecallKNN
                precisKNN = precisKNN ++ foldPrecisionKNN
                fScoreKNN = fScoreKNN ++ foldFScoreKNN

            } // for cv

            val formatHeader = "%10s %10s %10s %10s %10s\n"
            val formatMain   = "%-10s %10.9f %10.9f %10.9f %10.9f\n"

            val methodNM     = "NullModel"
            val methodNB     = "NaiveBayes"
            val methodTAN    = "TANBayes"
            val methodLDA    = "LDA"
            val methodLR     = "LR"
            val methodKNN    = "KNN"

            println ()
            println ("|====================|")
            println(s"|  $dataset DATASET  |")
            println(s"|     $nx   FOLDS     |")
            println ("|====================|")
            println ()

            println(">meanCV")
            printf(formatHeader, "", "accuracy", "recall", "precision", "f-score")
            printf(formatMain, methodNM,  accNM.mean,  recallNM.mean,  precisNM.mean,  fScoreNM.mean)
            printf(formatMain, methodNB,  accNB.mean,  recallNB.mean,  precisNB.mean,  fScoreNB.mean)
            printf(formatMain, methodTAN, accTAN.mean, recallTAN.mean, precisTAN.mean, fScoreTAN.mean)
            printf(formatMain, methodLDA, accLDA.mean, recallLDA.mean, precisLDA.mean, fScoreLDA.mean)
            printf(formatMain, methodLR,  accLR.mean,  recallLR.mean,  precisLR.mean,  fScoreLR.mean)
            printf(formatMain, methodKNN, accKNN.mean, recallKNN.mean, precisKNN.mean, fScoreKNN.mean)
            println("==========================================================")
            println()
            println(">stdCV")
            printf(formatHeader, "", "accuracy", "recall", "precision", "f-score")
            printf(formatMain, methodNM,  sqrt(accNM.variance),  sqrt(recallNM.variance),  sqrt(precisNM.variance),  sqrt(fScoreNM.variance))
            printf(formatMain, methodNB,  sqrt(accNB.variance),  sqrt(recallNB.variance),  sqrt(precisNB.variance),  sqrt(fScoreNB.variance))
            printf(formatMain, methodTAN, sqrt(accTAN.variance), sqrt(recallTAN.variance), sqrt(precisTAN.variance), sqrt(fScoreTAN.variance))
            printf(formatMain, methodLDA, sqrt(accLDA.variance), sqrt(recallLDA.variance), sqrt(precisLDA.variance), sqrt(fScoreLDA.variance))
            printf(formatMain, methodLR,  sqrt(accLR.variance),  sqrt(recallLR.variance),  sqrt(precisLR.variance),  sqrt(fScoreLR.variance))
            printf(formatMain, methodKNN, sqrt(accKNN.variance), sqrt(recallKNN.variance), sqrt(precisKNN.variance), sqrt(fScoreKNN.variance))
            println("==========================================================")
            println()

        } // crossValidateRand

  crossValidateRand2 (" TENNIS ", xyTennis,  xTennis,  yTennis,  fnTennis,  cnTennis,  10)
  crossValidateRand2 ("CARAVAN ", xyCaravan, xCaravan, yCaravan, fnCaravan, cnCaravan, 10, true)
  crossValidateRand2 (" CANCER ", xyCancer,  xCancer,  yCancer,  fnCancer,  cnCancer,  10)
 
  crossValidateRand2 (" TENNIS ", xyTennis,  xTennis,  yTennis,  fnTennis,  cnTennis,  14)              // LOOCV
  crossValidateRand2 ("CARAVAN ", xyCaravan, xCaravan, yCaravan, fnCaravan, cnCaravan, 20, true)
  crossValidateRand2 (" CANCER ", xyCancer,  xCancer,  yCancer,  fnCancer,  cnCancer,  20)

} // ClassifierTests
