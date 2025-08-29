package main

// Minimal EXIF parser for FocalLength (0x920A) as rational -> float64

import "encoding/binary"

func readEXIFFocalLengthJPEG(b []byte) (float64, bool) {
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
            if len(seg) >= 6 && seg[0] == 'E' && seg[1] == 'x' && seg[2] == 'i' && seg[3] == 'f' && seg[4] == 0 && seg[5] == 0 {
                if v, ok := parseTIFFFocal(seg[6:]); ok { return v, true }
            }
        }
        i += segLen - 2
    }
    return 0, false
}

func parseTIFFFocal(tiff []byte) (float64, bool) {
    if len(tiff) < 8 { return 0, false }
    le := false
    if tiff[0] == 'I' && tiff[1] == 'I' { le = true } else if !(tiff[0] == 'M' && tiff[1] == 'M') { return 0, false }
    u16 := func(off int) uint16 { if le { return binary.LittleEndian.Uint16(tiff[off:off+2]) }; return binary.BigEndian.Uint16(tiff[off:off+2]) }
    u32 := func(off int) uint32 { if le { return binary.LittleEndian.Uint32(tiff[off:off+4]) }; return binary.BigEndian.Uint32(tiff[off:off+4]) }
    if u16(2) != 42 { return 0, false }
    ifd0 := int(u32(4))
    if ifd0 <= 0 || ifd0+2 > len(tiff) { return 0, false }
    count := int(u16(ifd0))
    off := ifd0 + 2
    for i := 0; i < count; i++ {
        if off+12 > len(tiff) { break }
        tag := u16(off)
        typ := u16(off+2)
        _ = typ
        _ = u32(off + 4)
        valOff := int(u32(off + 8))
        if tag == 0x8769 { // EXIF IFD pointer
            exifOff := valOff
            if exifOff <= 0 || exifOff+2 > len(tiff) { break }
            ec := int(u16(exifOff))
            eo := exifOff + 2
            for j := 0; j < ec; j++ {
                if eo+12 > len(tiff) { break }
                etag := u16(eo)
                etyp := u16(eo + 2)
                ecnt := u32(eo + 4)
                evalOff := int(u32(eo + 8))
                if etag == 0x920A && etyp == 5 && ecnt >= 1 { // FocalLength, RATIONAL
                    if evalOff+8 <= len(tiff) {
                        num := u32(evalOff)
                        den := u32(evalOff + 4)
                        if den != 0 { return float64(num) / float64(den), true }
                    }
                }
                eo += 12
            }
        }
        off += 12
    }
    return 0, false
}
