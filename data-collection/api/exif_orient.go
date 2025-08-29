package main

import (
    "encoding/binary"
    "image"
)

// ApplyEXIFOrientationIfAny returns a new image with EXIF orientation applied if found; otherwise returns img.
func ApplyEXIFOrientationIfAny(jpegBytes []byte, img image.Image) image.Image {
    orient, ok := readJPEGEXIFOrientation(jpegBytes)
    if !ok || orient == 1 { return img }
    switch orient {
    case 2:
        return flipHorizontal(img)
    case 3:
        return rotate180(img)
    case 4:
        return flipVertical(img)
    case 5:
        return rotate90(flipHorizontal(img))
    case 6:
        return rotate90(img)
    case 7:
        return rotate270(flipHorizontal(img))
    case 8:
        return rotate270(img)
    default:
        return img
    }
}

// Minimal JPEG EXIF Orientation parser; returns (orientation, ok)
func readJPEGEXIFOrientation(b []byte) (int, bool) {
    if len(b) < 4 || b[0] != 0xFF || b[1] != 0xD8 { return 0, false }
    i := 2
    for i+4 < len(b) {
        if b[i] != 0xFF { return 0, false }
        marker := b[i+1]
        i += 2
        if marker == 0xD9 || marker == 0xDA { break }
        if i+2 > len(b) { break }
        segLen := int(b[i])<<8 | int(b[i+1])
        i += 2
        if segLen < 2 || i+segLen-2 > len(b) { break }
        if marker == 0xE1 {
            seg := b[i : i+segLen-2]
            if len(seg) >= 6 && string(seg[:6]) == "Exif\x00\x00" {
                return parseEXIFOrientation(seg[6:])
            }
        }
        i += segLen - 2
    }
    return 0, false
}

func parseEXIFOrientation(tiff []byte) (int, bool) {
    if len(tiff) < 8 { return 0, false }
    le := false
    if tiff[0] == 'I' && tiff[1] == 'I' { le = true } else if !(tiff[0] == 'M' && tiff[1] == 'M') { return 0, false }
    u16 := func(off int) uint16 { if le { return binary.LittleEndian.Uint16(tiff[off:off+2]) } ; return binary.BigEndian.Uint16(tiff[off:off+2]) }
    u32 := func(off int) uint32 { if le { return binary.LittleEndian.Uint32(tiff[off:off+4]) } ; return binary.BigEndian.Uint32(tiff[off:off+4]) }
    if u16(2) != 42 { return 0, false }
    ifd0 := int(u32(4))
    if ifd0 <= 0 || ifd0+2 > len(tiff) { return 0, false }
    count := int(u16(ifd0))
    off := ifd0 + 2
    for i := 0; i < count; i++ {
        if off+12 > len(tiff) { break }
        tag := u16(off)
        typ := u16(off+2)
        valOff := off + 8
        if tag == 0x0112 { // Orientation
            var val uint16
            if typ == 3 { // SHORT
                val = u16(valOff)
            } else { val = 1 }
            if val >= 1 && val <= 8 { return int(val), true }
            return 0, false
        }
        off += 12
    }
    return 0, false
}

// Simple image transforms
func rotate90(src image.Image) image.Image {
    b := src.Bounds(); w, h := b.Dx(), b.Dy()
    dst := image.NewRGBA(image.Rect(0, 0, h, w))
    for y := 0; y < h; y++ {
        for x := 0; x < w; x++ { dst.Set(h-1-y, x, src.At(b.Min.X+x, b.Min.Y+y)) }
    }
    return dst
}
func rotate180(src image.Image) image.Image {
    b := src.Bounds(); w, h := b.Dx(), b.Dy()
    dst := image.NewRGBA(image.Rect(0, 0, w, h))
    for y := 0; y < h; y++ {
        for x := 0; x < w; x++ { dst.Set(w-1-x, h-1-y, src.At(b.Min.X+x, b.Min.Y+y)) }
    }
    return dst
}
func rotate270(src image.Image) image.Image {
    b := src.Bounds(); w, h := b.Dx(), b.Dy()
    dst := image.NewRGBA(image.Rect(0, 0, h, w))
    for y := 0; y < h; y++ {
        for x := 0; x < w; x++ { dst.Set(y, w-1-x, src.At(b.Min.X+x, b.Min.Y+y)) }
    }
    return dst
}
func flipHorizontal(src image.Image) image.Image {
    b := src.Bounds(); w, h := b.Dx(), b.Dy()
    dst := image.NewRGBA(image.Rect(0, 0, w, h))
    for y := 0; y < h; y++ {
        for x := 0; x < w; x++ { dst.Set(w-1-x, y, src.At(b.Min.X+x, b.Min.Y+y)) }
    }
    return dst
}
func flipVertical(src image.Image) image.Image {
    b := src.Bounds(); w, h := b.Dx(), b.Dy()
    dst := image.NewRGBA(image.Rect(0, 0, w, h))
    for y := 0; y < h; y++ {
        for x := 0; x < w; x++ { dst.Set(x, h-1-y, src.At(b.Min.X+x, b.Min.Y+y)) }
    }
    return dst
}
