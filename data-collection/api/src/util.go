package main

import (
    "bytes"
    "crypto/rand"
    "encoding/hex"
    "image"
    _ "image/gif"
    "image/jpeg"
    "image/png"
    "io"
)

func newID() string {
    b := make([]byte, 16)
    _, _ = rand.Read(b)
    return hex.EncodeToString(b)
}

func jpegEncode(w io.Writer, img image.Image) error {
    return jpeg.Encode(w, img, &jpeg.Options{Quality: 90})
}

func pngEncode(w io.Writer, img image.Image) error {
    enc := png.Encoder{CompressionLevel: png.BestCompression}
    return enc.Encode(w, img)
}

// reencodeToStripMetadata decodes and re-encodes the image to drop metadata.
// Best-effort: returns nil if decode fails.
func reencodeToStripMetadata(b []byte) ([]byte, string) {
    img, format, err := image.Decode(bytes.NewReader(b))
    if err != nil { return nil, "" }
    var enc bytes.Buffer
    switch format {
    case "png":
        if err := pngEncode(&enc, img); err != nil { return nil, "" }
        return enc.Bytes(), ".png"
    case "jpeg":
        // Apply EXIF orientation normalization if present
        norm := ApplyEXIFOrientationIfAny(b, img)
        if err := jpegEncode(&enc, norm); err != nil { return nil, "" }
        return enc.Bytes(), ".jpg"
    default:
        if err := jpegEncode(&enc, img); err != nil { return nil, "" }
        return enc.Bytes(), ".jpg"
    }
}

