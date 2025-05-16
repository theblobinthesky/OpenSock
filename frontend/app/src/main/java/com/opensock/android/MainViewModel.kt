package com.opensock.android

import android.net.Uri
import androidx.compose.runtime.State
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.core.net.toUri
import androidx.lifecycle.ViewModel

enum class InstanceSelectorState {
    INITIAL, CLICKED, WAITING, READY_TO_SAVE
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

    private val _selectedMode = mutableIntStateOf(0)
    val selectedMode: State<Int> = _selectedMode

    private val _instanceSelectorState = mutableStateOf(InstanceSelectorState.INITIAL)
    val instanceSelectorState: State<InstanceSelectorState> = _instanceSelectorState

    private val _selectedInstanceIdx = mutableIntStateOf(0)
    val selectedInstanceIdx: State<Int> = _selectedInstanceIdx

    private val _selectedOwner = mutableIntStateOf(0)
    val selectedOwner: State<Int> = _selectedOwner

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

    fun setSelectedMode(mode: Int) {
        _selectedMode.value = mode
    }

    fun setInstanceSelectorState(state: InstanceSelectorState) {
        _instanceSelectorState.value = state
    }

    fun clickInstance(instances: Instances, idx: Int) {
        if (isInstanceSelected(instances, idx)) {
            _instanceSelectorState.value = InstanceSelectorState.WAITING
        } else {
            _instanceSelectorState.value = InstanceSelectorState.CLICKED
            _selectedInstanceIdx.intValue = idx
        }
    }

    fun isInstanceSelected(instances: Instances, idx: Int): Boolean {
        val otherIdx = instances.getOtherIdx(idx)
        return instanceSelectorState.value == InstanceSelectorState.CLICKED
                && (selectedInstanceIdx.value == idx || selectedInstanceIdx.value == otherIdx)
    }

    fun setSelectedOwner(owner: Int) {
        _selectedOwner.value = owner
    }

}