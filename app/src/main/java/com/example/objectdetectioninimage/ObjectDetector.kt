package com.example.android.camera.utils.com.example.trafficlightdetection

import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import android.util.Size
import com.example.objectdetectioninimage.DetectionObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp


/**
 * CameraXの物体検知の画像解析ユースケース
 * @param interpreter tfliteモデルを操作するライブラリ
 * @param labels 正解ラベルのリスト
 * @param resultViewSize 結果を表示するsurfaceViewのサイズ
 * @param listener コールバックで解析結果のリストを受け取る
 */
class ObjectDetector(
    private val interpreter: Interpreter,
    private val labels: List<String>,
    private val resultViewSize: Size
){

    companion object {
        // モデルのinputとoutputサイズ
        private const val IMG_SIZE_X = 300
        private const val IMG_SIZE_Y = 300
        private const val MAX_DETECTION_NUM = 10

        // 今回使うtfliteモデルは量子化済みなのでnormalize関連は127.5fではなく以下の通り
        private const val NORMALIZE_MEAN = 0f
        private const val NORMALIZE_STD = 1f

        // ===== 適宜変更 =====
        // 検出結果のスコアしきい値
        private const val SCORE_THRESHOLD = 0.5f
    }

    fun analyze(bitmap: Bitmap): List<DetectionObject>  {
        // ===== ここで推論処理 =====
        return detect(bitmap)
    }

    private val tfImageProcessor by lazy {

        ImageProcessor.Builder()
            // Center crop the image to the largest square possible
            //.add(ResizeWithCropOrPadOp(size, size)) // 入力画像をクロップ
            .add(ResizeOp(IMG_SIZE_X, IMG_SIZE_Y, ResizeOp.ResizeMethod.BILINEAR)) // モデルのinputに合うように画像のリサイズ
            .add(NormalizeOp(NORMALIZE_MEAN, NORMALIZE_STD)) // normalization関連
            .build()
    }

    private val tfImageBuffer = TensorImage(DataType.UINT8)

    // 検出結果のバウンディングボックス [1:10:4]
    // バウンディングボックスは [top, left, bottom, right] の形
    private val outputBoundingBoxes: Array<Array<FloatArray>> = arrayOf(
        Array(MAX_DETECTION_NUM) {
            FloatArray(4)
        }
    )

    // 検出結果のクラスラベルインデックス [1:10]
    private val outputLabels: Array<FloatArray> = arrayOf(
        FloatArray(MAX_DETECTION_NUM)
    )

    // 検出結果の各スコア [1:10]
    private val outputScores: Array<FloatArray> = arrayOf(
        FloatArray(MAX_DETECTION_NUM)
    )

    // 検出した物体の数(今回はtflite変換時に設定されているので 10 (一定))
    private val outputDetectionNum: FloatArray = FloatArray(1)

    // 検出結果を受け取るためにmapにまとめる
    private val outputMap = mapOf(
        0 to outputBoundingBoxes,
        1 to outputLabels,
        2 to outputScores,
        3 to outputDetectionNum
    )

    // ===== 推論処理 =====
    // 画像をRGB bitmap -> tensorflowImage -> tensorflowBufferに変換して推論し結果をリストとして出力
    // 引数をbitmap形式に変更
    private fun detect(targetBitmap: Bitmap): List<DetectionObject> {
        //val targetBitmap = Bitmap.createBitmap(targetImage.width, targetImage.height, Bitmap.Config.ARGB_8888)
        tfImageBuffer.load(targetBitmap)
        val tensorImage = tfImageProcessor.process(tfImageBuffer)

        //tfliteモデルで推論の実行
        interpreter.runForMultipleInputsOutputs(arrayOf(tensorImage.buffer), outputMap)

        // 推論結果を整形してリストにして返す
        val detectedObjectList = arrayListOf<DetectionObject>()
        loop@ for (i in 0 until outputDetectionNum[0].toInt()) {
            val score = outputScores[0][i]
            val label = labels[outputLabels[0][i].toInt()]
            val boundingBox = RectF(
                outputBoundingBoxes[0][i][1] * resultViewSize.width,
                outputBoundingBoxes[0][i][0] * resultViewSize.height,
                outputBoundingBoxes[0][i][3] * resultViewSize.width,
                outputBoundingBoxes[0][i][2] * resultViewSize.height
            )

            // しきい値よりも大きいもののみ追加
            if ( score >= ObjectDetector.SCORE_THRESHOLD) {
                detectedObjectList.add(
                    DetectionObject(
                        score = score,
                        label = label,
                        boundingBox = boundingBox
                    )
                )
                Log.d("デバッグ", "検出:" + label + " : " + score )
            } else {
                // 検出結果はスコアの高い順にソートされたものが入っているので、しきい値を下回ったらループ終了
                break@loop
            }
        }
        return detectedObjectList.take(4)
    }
}