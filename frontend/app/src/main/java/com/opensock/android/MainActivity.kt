package com.opensock.android

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.absoluteOffset
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentWidth
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Check
import androidx.compose.material.icons.rounded.Person
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.opensock.android.ui.theme.OpenSockAndroidAppTheme
import androidx.core.net.toUri
import androidx.lifecycle.compose.LocalLifecycleOwner
import coil3.compose.AsyncImage
import java.io.File
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

private const val TAG = "OpenSock"

class MainActivity : ComponentActivity() {
    private val viewModel: MainViewModel by viewModels()

    private val cameraPermissionRequest =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            viewModel.setPermissionGranted(isGranted)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (ContextCompat.checkSelfPermission(
                this, Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            viewModel.setPermissionGranted(true)
        } else {
            cameraPermissionRequest.launch(Manifest.permission.CAMERA)
        }

        enableEdgeToEdge()
        setContent {
            OpenSockAndroidAppTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    if (viewModel.isPermissionGranted.value) {
                        val instances = Instances(
                            instances = listOf(
                                Instance(0.2f, 0.4f, 0),
                                Instance(1.2f, 1.4f, 0),
                                Instance(0.7f, 0.9f, 1),
                                Instance(1.9f, 1.4f, 1),
                            ), pairs = listOf(
                                Pair(0, 1), Pair(2, 3)
                            ), owners = listOf(
                                "Magda",
                                "Erik",
                                "Mudah",
                                "Mudah",
                                "Mudah",
                                "Mudah",
                                "Mudah",
                                "Mudah"
                            )
                        )

                        if (viewModel.numPicturesTaken.value < 2) {
                            CameraPreviewScreen(viewModel)
                        } else if (viewModel.selectedMode.value == 0) {
                            InstancePairingScreen(instances, viewModel)
                        } else {
                            InstanceOwnerScreen(instances, viewModel)
                        }
                    } else {
                        CameraPermissionRequiredScreen(context = this)
                    }
                }
            }
        }
    }
}

private suspend fun Context.getCameraProvider(): ProcessCameraProvider =
    suspendCoroutine { continuation ->
        ProcessCameraProvider.getInstance(this).also { cameraProvider ->
            cameraProvider.addListener({
                continuation.resume(cameraProvider.get())
            }, ContextCompat.getMainExecutor(this))
        }
    }

@Composable
fun CameraPreviewScreen(viewModel: MainViewModel) {
    val lensFacing = CameraSelector.LENS_FACING_BACK
    val lifecycleOwner = LocalLifecycleOwner.current
    val context = LocalContext.current
    val preview = Preview.Builder().build()
    val previewView = remember {
        PreviewView(context)
    }
    val imageCapture = remember {
        ImageCapture.Builder().build()
    }
    val cameraxSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
    LaunchedEffect(lensFacing) {
        val cameraProvider = context.getCameraProvider()
        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(lifecycleOwner, cameraxSelector, preview)
        cameraProvider.bindToLifecycle(lifecycleOwner, cameraxSelector, imageCapture)
        preview.surfaceProvider = previewView.surfaceProvider
    }

    ImageLayout(bgContent = {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxWidth())
    }, bottomContent = {
        ModeSelectorBar(
            listOf("Find Pairs", "Assign Socks"), viewModel.selectedMode.value,
            onModeSelected = { viewModel.setSelectedMode(it) },
        )

        ShutterButton(viewModel, imageCapture)
    })
}

@Composable
fun ImageLayout(
    bgContent: @Composable BoxScope.() -> Unit, bottomContent: @Composable ColumnScope.() -> Unit
) {
    Box(modifier = Modifier.fillMaxSize()) {
        bgContent()

        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 36.dp, start = 12.dp, end = 12.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            content = bottomContent
        )
    }
}

@Composable
fun ShutterButton(viewModel: MainViewModel, imageCapture: ImageCapture) {
    val context = LocalContext.current

    Box(
        modifier = Modifier
            .size(120.dp)
            .clip(CircleShape)
            .clickable(
                onClick = {
                    val outputFileOptions = ImageCapture.OutputFileOptions.Builder(
                        File(
                            context.cacheDir, "photo_${System.currentTimeMillis()}.jpg"
                        )
                    ).build()

                    imageCapture.takePicture(
                        outputFileOptions,
                        ContextCompat.getMainExecutor(context),
                        object : ImageCapture.OnImageSavedCallback {
                            override fun onError(error: ImageCaptureException) {
                                Log.w(TAG, error.toString())
                            }

                            override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                                viewModel.setPictureUri(outputFileResults.savedUri!!)
                                viewModel.setNumPicturesTaken(viewModel.numPicturesTaken.value + 1)
                            }
                        })
                })
    ) {
        Image(
            painter = painterResource(id = R.drawable.camera_shutter),
            contentDescription = null,
            colorFilter = ColorFilter.tint(Color.White),
            modifier = Modifier.size(120.dp)
        )
    }
}

@Composable
fun CameraPermissionRequiredScreen(context: Context) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Camera Permission Needed", style = MaterialTheme.typography.headlineMedium
        )
        Spacer(Modifier.height(16.dp))
        Text(
            text = "You must allow camera access to use this app. Please grant the permission in settings."
        )
        Spacer(Modifier.height(32.dp))
        Button(onClick = {
            val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                data = "package:${context.packageName}".toUri()
            }
            context.startActivity(intent)
        }) {
            Text("Go to Settings")
        }
        Spacer(Modifier.height(16.dp))
    }
}

@Composable
fun ModeSelectorBar(
    modes: List<String>,
    selectedIndex: Int,
    onModeSelected: (Int) -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        horizontalArrangement = Arrangement.Center,
        modifier = modifier
            .height(48.dp)
            .background(Color(0x22000000), shape = RoundedCornerShape(24.dp))
            .padding(4.dp)
    ) {
        modes.forEachIndexed { idx, mode ->
            val selected = idx == selectedIndex
            Box(
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(20.dp))
                    .background(if (selected) Color.White else Color.Transparent)
                    .clickable { onModeSelected(idx) }
                    .padding(vertical = 8.dp, horizontal = 16.dp),
                contentAlignment = Alignment.Center) {
                Text(
                    text = mode,
                    color = if (selected) MaterialTheme.colorScheme.primary else Color.White,
                    style = MaterialTheme.typography.titleMedium
                )
            }
        }
    }
}

@Composable
fun InstancePairingScreen(instances: Instances, viewModel: MainViewModel) {
    ImageLayout(bgContent = {
        AsyncImage(
            model = viewModel.pictureUri.value,
            contentDescription = null,
            modifier = Modifier.fillMaxSize()
        )
    }, bottomContent = {
        when (viewModel.instanceSelectorState.value) {
            InstanceSelectorState.INITIAL -> Text("Please select a sock to start...")

            InstanceSelectorState.CLICKED -> RejectAcceptButtons(onReject = {
                viewModel.setInstanceSelectorState(InstanceSelectorState.WAITING)
            }, onAccept = {
                viewModel.setInstanceSelectorState(InstanceSelectorState.WAITING)
            })

            InstanceSelectorState.WAITING -> UndoButton()

            InstanceSelectorState.READY_TO_SAVE -> UndoSaveButtons()
        }
    })

    instances.instances.forEachIndexed { index, inst ->
        val isSelected = viewModel.isInstanceSelected(instances, index)

        Box(
            modifier = Modifier
                .offset(
                    x = (inst.x * 150.0).dp, y = (inst.y * 200.0).dp
                )
                .size(64.dp)
                .clip(CircleShape)
                .background(
                    if (isSelected) Color.Yellow.copy(0.7f) else Color.Gray.copy(0.7f)
                )
                .clickable {
                    viewModel.clickInstance(instances, index)
                }, contentAlignment = Alignment.Center
        ) { }
    }
}

@Composable
fun InstanceOwnerScreen(instances: Instances, viewModel: MainViewModel) {
    val configuration = LocalConfiguration.current

    Column(Modifier.fillMaxSize()) {
        /* ---------- preview + clickable blobs ---------- */
        Box(
            Modifier
                .weight(1f)
                .fillMaxWidth() // Ensure it takes full width within the weighted space
        ) {
            AsyncImage(
                model = viewModel.pictureUri.value,
                contentDescription = "Preview Image", // Added for accessibility
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop // Or ContentScale.Fit, as appropriate
            )
            // Iterating over instances to draw blobs
            instances.instances.forEach { inst -> // Index not used, so forEach is simpler
                val selected = inst.owner == viewModel.selectedOwner.value
                Box(Modifier
                    .absoluteOffset { // place at % coordinates
                        IntOffset(
                            x = (inst.x * 150.0).toInt(), y = (inst.y * 200.0).toInt()
                        )
                    }
                    .size(64.dp)
                    .clip(CircleShape)
                    .background(
                        if (selected) Color.Yellow.copy(alpha = .7f)
                        else Color.Gray.copy(alpha = .7f)
                    )
                    // contentAlignment is not needed for an empty Box
                )
            }
        }

        Surface(
            modifier = Modifier.fillMaxWidth(),
            color = MaterialTheme.colorScheme.surfaceVariant,
            tonalElevation = 2.dp,
            shadowElevation = 2.dp
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 16.dp, horizontal = 20.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Rounded.Person, // Corrected usage
                    contentDescription = "Owner instruction icon", // Added for accessibility
                    tint = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.width(12.dp))
                Text(
                    text = "Choose an owner below", // Instruction text is clear
                    textAlign = TextAlign.Start,
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(max = configuration.screenHeightDp.dp / 3)
                .navigationBarsPadding()
        ) {
            LazyColumn(
                modifier = Modifier.fillMaxWidth(),
                contentPadding = PaddingValues(vertical = 12.dp, horizontal = 16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                itemsIndexed(instances.owners) { idx, name ->
                    val selected = viewModel.selectedOwner.value == idx

                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .animateContentSize()
                            .clickable { viewModel.setSelectedOwner(idx) },
                        shape = RoundedCornerShape(12.dp),
                        elevation = CardDefaults.cardElevation(
                            defaultElevation = if (selected) 6.dp else 2.dp
                        ),
                        colors = CardDefaults.cardColors(
                            containerColor = if (selected) MaterialTheme.colorScheme.primaryContainer
                            else MaterialTheme.colorScheme.surfaceVariant
                        )
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(36.dp)
                                    .clip(CircleShape)
                                    .background(
                                        if (selected) MaterialTheme.colorScheme.primary
                                        else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.7f)
                                    )
                                    .border(
                                        width = 1.dp,
                                        color = if (selected) MaterialTheme.colorScheme.primary
                                        else MaterialTheme.colorScheme.outline,
                                        shape = CircleShape
                                    ), contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = name.firstOrNull()?.toString()?.uppercase() ?: "?",
                                    style = MaterialTheme.typography.titleMedium,
                                    color = if (selected) MaterialTheme.colorScheme.onPrimary
                                    else MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }

                            Spacer(modifier = Modifier.width(16.dp))

                            // Owner name
                            Text(
                                text = name,
                                style = MaterialTheme.typography.titleMedium,
                                color = if (selected) MaterialTheme.colorScheme.onPrimaryContainer
                                else MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier.weight(1f) // Allow name to take available space and push icon
                            )

                            // Selection indicator
                            if (selected) {
                                Icon(
                                    imageVector = Icons.Rounded.Check,
                                    contentDescription = "Selected owner indicator",
                                    tint = MaterialTheme.colorScheme.primary
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun RejectAcceptButtons(
    onReject: () -> Unit = {}, onAccept: () -> Unit = {}
) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(32.dp),
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .wrapContentWidth()
            .height(120.dp)
    ) {
        ActionCircleButton(
            iconRes = R.drawable.close,
            contentDescription = "Reject",
            tint = Color.Red,
            onClick = onReject
        )
        ActionCircleButton(
            iconRes = R.drawable.check,
            contentDescription = "Accept",
            tint = Color(0xFF00C853), // green
            onClick = onAccept
        )
    }
}

@Composable
fun UndoButton(
    onUndo: () -> Unit = {}
) {
    ActionCircleButton(
        iconRes = R.drawable.arrow_back,
        contentDescription = null,
        tint = Color.Black,
        onClick = onUndo
    )
}

@Composable
fun UndoSaveButtons(
    onUndo: () -> Unit = {}, onSave: () -> Unit = {}
) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(32.dp),
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .wrapContentWidth()
            .height(120.dp)
    ) {
        ActionCircleButton(
            iconRes = R.drawable.arrow_back,
            contentDescription = "Undo",
            tint = Color.Red,
            onClick = onUndo
        )
        ActionCircleButton(
            iconRes = R.drawable.check,
            contentDescription = "Save",
            tint = Color(0xFF00C853), // green
            onClick = onSave
        )
    }
}

@Composable
fun ActionCircleButton(
    iconRes: Int, contentDescription: String?, tint: Color, onClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .size(96.dp)
            .clip(CircleShape)
            .background(Color.White)
            .clickable(onClick = onClick), contentAlignment = Alignment.Center
    ) {
        Image(
            painter = painterResource(id = iconRes),
            contentDescription = contentDescription,
            colorFilter = ColorFilter.tint(tint),
            modifier = Modifier.size(56.dp)
        )
    }
}
