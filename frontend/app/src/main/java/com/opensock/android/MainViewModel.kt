package com.opensock.android

import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel

class MainViewModel : ViewModel() {
    private val _isPermissionGranted = mutableStateOf(false)
    val isPermissionGranted: State<Boolean> = _isPermissionGranted

    fun setPermissionGranted(granted: Boolean) {
        _isPermissionGranted.value = granted
    }
}