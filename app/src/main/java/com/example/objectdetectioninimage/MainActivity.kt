package com.example.objectdetectioninimage

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.res.AssetFileDescriptor
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.android.camera.utils.com.example.trafficlightdetection.ObjectDetector
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.channels.FileChannel


class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        button.setOnClickListener{ selectPhoto() }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, resultData: Intent?) {
        super.onActivityResult(requestCode, resultCode, resultData)
        if (resultCode != RESULT_OK) {
            return
        }
        when (requestCode) {
            READ_REQUEST_CODE -> {
                try {
                    resultData?.data?.also { uri ->
                        val inputStream = contentResolver?.openInputStream(uri)
                        //既にBitmap形式
                        val image = BitmapFactory.decodeStream(inputStream)

                        /**
                         * 推論処理
                         */
                        val detectedObjectList = ObjectDetector(
                            interpreter,
                            labels,
                            Size(image.width, image.height)
                        ).analyze(image)

                        Log.d("デバッグ", "内容:" + detectedObjectList)

                        // ===== 検出結果の表示 =====
                        val myView = MyView(this, image, detectedObjectList,
                            Size(layout.width, layout.height))
                        myView.setOnClickListener{ selectPhoto() }
                        setContentView(myView)

                        Log.d("デバッグ", "Draw")

                    }
                } catch (e: Exception) {
                    Toast.makeText(this, "エラーが発生しました", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun selectPhoto() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "image/*"
        }
        startActivityForResult(intent, READ_REQUEST_CODE)
    }

    // ===== Tensorflow Lite で使うために追加 =====
    // tfliteモデルを扱うためのラッパーを含んだinterpreter
    private val interpreter: Interpreter by lazy {
        Interpreter(loadModel())
    }

    // モデルの正解ラベルリスト
    private val labels: List<String> by lazy {
        loadLabels()
    }

    // tfliteモデルをassetsから読み込む
    private fun loadModel(fileName: String = MODEL_FILE_NAME): ByteBuffer {
        lateinit var modelBuffer: ByteBuffer
        var file: AssetFileDescriptor? = null
        try {
            file = assets.openFd(fileName)
            val inputStream = FileInputStream(file.fileDescriptor)
            val fileChannel = inputStream.channel
            modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, file.startOffset, file.declaredLength)
        } catch (e: Exception) {
            Toast.makeText(this, "モデルファイル読み込みエラー", Toast.LENGTH_SHORT).show()
            finish()
        } finally {
            file?.close()
        }
        return modelBuffer
    }

    // モデルの正解ラベルデータをassetsから取得
    private fun loadLabels(fileName: String = MainActivity.LABEL_FILE_NAME): List<String> {
        var labels = listOf<String>()
        var inputStream: InputStream? = null
        try {
            inputStream = assets.open(fileName)
            val reader = BufferedReader(InputStreamReader(inputStream))
            labels = reader.readLines()
        } catch (e: Exception) {
            Toast.makeText(this, "txtファイル読み込みエラー", Toast.LENGTH_SHORT).show()
            finish()
        } finally {
            inputStream?.close()
        }
        return labels
    }

    // Viewを継承したクラス
    class MyView(context: Context, bitmap: Bitmap, detectedObjectList: List<DetectionObject>,
                 layout: Size) : View(context) {

        private var paint: Paint = Paint()

        init {
        }

        private val layout = layout
        private val bmp = bitmap
        private val dol = detectedObjectList
        private val pathColorList = listOf(Color.RED, Color.GREEN, Color.CYAN, Color.BLUE)

        @SuppressLint("DrawAllocation")
        override fun onDraw(canvas: Canvas){

            // drawBitmapを使って取得画像を描画
            val scale = layout.width.toFloat() / bmp.width
            canvas.scale(scale, scale)
            canvas.drawBitmap(bmp, 0f, 10f, paint)

            // 確認画像と詳細の表示
            val previewBmp = Bitmap.createBitmap(
                bmp,
                bmp.width/2 - bmp.width/4,
                bmp.height/2 - bmp.height/4,
                bmp.width/2,
                bmp.height/2,
                null,
                true
            )
            canvas.drawBitmap(bmp, 0f, bmp.height + 20f, paint)
            paint.apply {
                color = Color.MAGENTA
                style = Paint.Style.FILL
                isAntiAlias = true
                textSize = bmp.height / 12f
            }
            // drawTextを使って文字を描画
            canvas.drawText(
                "Width * Height : " + bmp.width + " * " + bmp.height,
                10f,
                2 * bmp.height + 20f,
                paint
            )

            paint.apply {
                color = Color.GRAY
                style = Paint.Style.FILL_AND_STROKE
                isAntiAlias = true
                textSize = bmp.height / 10f
            }
            canvas.drawText(
                "[ Touch to change the image. ]",
                100f,
                2 * bmp.height + 300f,
                paint
            )

            // indicesでインデックスを取得する
            for( i in dol.indices){

                paint.apply {
                    color = pathColorList[i]
                    style = Paint.Style.STROKE
                    strokeWidth = 5f
                    isAntiAlias = false
                }

                // drawRectを使って矩形を描画する、引数に座標を設定
                // (x1,y1,x2,y2,paint) 左上の座標(x1,y1), 右下の座標(x2,y2)
                canvas.drawRect(dol[i].boundingBox, paint)

                // ラベルとスコアの表示
                paint.apply {
                    style = Paint.Style.FILL
                    isAntiAlias = true
                    textSize = bmp.height / 12f
                }

                // drawTextを使って文字を描画
                canvas.drawText(
                    dol[i].label + " " + "%,.2f".format(dol[i].score * 100) + "%",
                    dol[i].boundingBox.left,
                    dol[i].boundingBox.top - 5f,
                    paint
                )

            }
        }
    }

    companion object {
        private const val READ_REQUEST_CODE: Int = 42

        // モデル名とラベル名
        private const val MODEL_FILE_NAME = "ssd_mobilenet_v1_1_metadata_1.tflite"
        private const val LABEL_FILE_NAME = "coco_dataset_labels.txt"
    }
}