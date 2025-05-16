package com.opensock.android

import android.net.Uri
import androidx.compose.runtime.State
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.core.net.toUri
import androidx.lifecycle.ViewModel

enum class InstanceSelectorState {
    INITIAL,
    CLICKED,
    WAITING,
    READY_TO_SAVE
}

class MainViewModel : ViewModel() {
    private val _isPermissionGranted = mutableStateOf(false)
    val isPermissionGranted: State<Boolean> = _isPermissionGranted

    private val _numPicturesTaken = mutableIntStateOf(0)
    val numPicturesTaken: State<Int> = _numPicturesTaken

    private val _pictureUri = mutableStateOf("".toUri())
    val pictureUri: State<Uri> = _pictureUri

    private val _pictureUri2 = mutableStateOf("".toUri())
    val pictureUri2: State<Uri> = _pictureUri2

    private val _instanceSelectorState = mutableStateOf(InstanceSelectorState.INITIAL)
    val instanceSelectorState: State<InstanceSelectorState> = _instanceSelectorState

    fun setPermissionGranted(granted: Boolean) {
        _isPermissionGranted.value = granted
    }

    fun setNumPicturesTaken(numPicturesTaken: Int) {
        _numPicturesTaken.intValue = numPicturesTaken
    }

    fun setPictureUri(pictureUri: Uri) {
        if (numPicturesTaken.value == 0) {
            _pictureUri.value = pictureUri
        } else {
            _pictureUri2.value = pictureUri
        }
    }

    fun setInstanceSelectorState(state: InstanceSelectorState) {
        _instanceSelectorState.value = state
    }

}