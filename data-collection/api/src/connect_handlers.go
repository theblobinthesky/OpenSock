package main

import (
    "bytes"
    "context"
    "errors"
    "net/http"
    "time"
    "connectrpc.com/connect"
    dcv1 "opensock/datacollection/proto/gen/opensock/dc/v1"
    dcv1connect "opensock/datacollection/proto/gen/opensock/dc/v1/dcv1connect"
)

// DataCollection Service
type ConnectServer struct{ store Store }

func (s *ConnectServer) CreateSession(ctx context.Context, req *connect.Request[dcv1.CreateSessionRequest]) (*connect.Response[dcv1.CreateSessionResponse], error) {
    r := req.Msg
    cs := CreateSessionRequest{
        Name: r.GetName(), Handle: r.GetHandle(), Email: r.GetEmail(),
        NotifyOptIn: r.GetNotifyOptIn(), Language: r.GetLanguage(), Mode: r.GetMode().String(),
    }
    if cs.Email == "" {
        return nil, connect.NewError(connect.CodeInvalidArgument, errors.New("email required"))
    }
    sess, err := s.store.CreateSession(cs)
    if err != nil { return nil, connect.NewError(connect.CodeInvalidArgument, err) }
    return connect.NewResponse(&dcv1.CreateSessionResponse{SessionId: sess.ID}), nil
}

func (s *ConnectServer) FinalizeSession(ctx context.Context, req *connect.Request[dcv1.FinalizeSessionRequest]) (*connect.Response[dcv1.FinalizeSessionResponse], error) {
    if err := s.store.FinalizeSession(req.Msg.GetSessionId()); err != nil {
        return nil, connect.NewError(connect.CodeInvalidArgument, err)
    }
    return connect.NewResponse(&dcv1.FinalizeSessionResponse{Ok: true}), nil
}

// Admin Service
type AdminServer struct{ store Store }

func (s *AdminServer) ListSessions(ctx context.Context, req *connect.Request[dcv1.ListSessionsRequest]) (*connect.Response[dcv1.ListSessionsResponse], error) {
    sessions := s.store.SessionsQuery(req.Msg.GetEmail(), req.Msg.GetName(), req.Msg.GetMode(), req.Msg.GetStatus())
    out := &dcv1.ListSessionsResponse{}
    for _, ss := range sessions {
        var fin string
        if ss.FinalizedAt != nil { fin = ss.FinalizedAt.UTC().Format(time.RFC3339) }
        out.Sessions = append(out.Sessions, &dcv1.SessionSummary{
            Id: ss.ID,
            Email: ss.Email,
            Name: ss.Name,
            Handle: ss.Handle,
            Mode: ss.Mode,
            ImageCount: int32(ss.ImageCount),
            CreatedAt: ss.CreatedAt.UTC().Format(time.RFC3339),
            FinalizedAt: fin,
        })
    }
    return connect.NewResponse(out), nil
}

func (s *AdminServer) DeleteSession(ctx context.Context, req *connect.Request[dcv1.DeleteSessionRequest]) (*connect.Response[dcv1.DeleteResponse], error) {
    if err := s.store.DeleteSession(req.Msg.GetSessionId()); err != nil {
        return nil, connect.NewError(connect.CodeInvalidArgument, err)
    }
    return connect.NewResponse(&dcv1.DeleteResponse{Ok: true}), nil
}

func (s *AdminServer) DeleteUser(ctx context.Context, req *connect.Request[dcv1.DeleteUserRequest]) (*connect.Response[dcv1.DeleteResponse], error) {
    if err := s.store.DeleteUser(req.Msg.GetEmail()); err != nil {
        return nil, connect.NewError(connect.CodeInvalidArgument, err)
    }
    return connect.NewResponse(&dcv1.DeleteResponse{Ok: true}), nil
}

func (s *AdminServer) GetConfig(ctx context.Context, req *connect.Request[dcv1.GetAdminConfigRequest]) (*connect.Response[dcv1.GetAdminConfigResponse], error) {
    cap, minMixed, minSame, maxMB, err := s.store.GetConfig()
    if err != nil { return nil, connect.NewError(connect.CodeInvalidArgument, err) }
    return connect.NewResponse(&dcv1.GetAdminConfigResponse{SessionImageCap: uint32(cap), MinRequiredMixed: uint32(minMixed), MinRequiredSame: uint32(minSame), FileSizeLimitMb: uint32(maxMB)}), nil
}

func (s *AdminServer) UpdateConfig(ctx context.Context, req *connect.Request[dcv1.UpdateAdminConfigRequest]) (*connect.Response[dcv1.GetAdminConfigResponse], error) {
    if err := s.store.SetConfig(int(req.Msg.GetSessionImageCap()), int(req.Msg.GetMinRequiredMixed()), int(req.Msg.GetMinRequiredSame()), int(req.Msg.GetFileSizeLimitMb())); err != nil {
        return nil, connect.NewError(connect.CodeInvalidArgument, err)
    }
    cap, minMixed, minSame, maxMB, _ := s.store.GetConfig()
    return connect.NewResponse(&dcv1.GetAdminConfigResponse{SessionImageCap: uint32(cap), MinRequiredMixed: uint32(minMixed), MinRequiredSame: uint32(minSame), FileSizeLimitMb: uint32(maxMB)}), nil
}

// Upload Service
type UploadServer struct{ store Store }

func (s *UploadServer) InitUpload(ctx context.Context, req *connect.Request[dcv1.InitUploadRequest]) (*connect.Response[dcv1.InitUploadResponse], error) {
    id, err := s.store.InitUpload(req.Msg.GetSessionId(), req.Msg.GetFilename(), req.Msg.GetMimeType(), int64(req.Msg.GetTotalSize()))
    if err != nil { return nil, connect.NewError(connect.CodeInvalidArgument, err) }
    return connect.NewResponse(&dcv1.InitUploadResponse{UploadId: id}), nil
}

func (s *UploadServer) UploadChunk(ctx context.Context, req *connect.Request[dcv1.UploadChunkRequest]) (*connect.Response[dcv1.UploadChunkResponse], error) {
    if err := s.store.UploadChunk(req.Msg.GetUploadId(), int(req.Msg.GetIndex()), bytes.NewReader(req.Msg.GetChunk())); err != nil {
        return nil, connect.NewError(connect.CodeInvalidArgument, err)
    }
    return connect.NewResponse(&dcv1.UploadChunkResponse{Ok: true}), nil
}

func (s *UploadServer) CompleteUpload(ctx context.Context, req *connect.Request[dcv1.CompleteUploadRequest]) (*connect.Response[dcv1.CompleteUploadResponse], error) {
    sid, _, err := s.store.CompleteUpload(req.Msg.GetUploadId())
    if err != nil { return nil, connect.NewError(connect.CodeInvalidArgument, err) }
    return connect.NewResponse(&dcv1.CompleteUploadResponse{Ok: true, SessionId: sid}), nil
}

// Config Service
type ConfigServer struct{ cfg Config; store Store }

func (s *ConfigServer) Get(ctx context.Context, req *connect.Request[dcv1.GetConfigRequest]) (*connect.Response[dcv1.GetConfigResponse], error) {
    cap := s.cfg.SessionImageCap
    minMixed, minSame, maxMB := 4, 3, 25
    if s.store != nil { if c, mm, ms, mb, err := s.store.GetConfig(); err == nil { cap, minMixed, minSame, maxMB = c, mm, ms, mb } }
    return connect.NewResponse(&dcv1.GetConfigResponse{ SessionImageCap: uint32(cap), MinRequired: map[string]uint32{"MIXED_UNIQUES": uint32(minMixed), "SAME_TYPE": uint32(minSame)}, FileSizeLimitMb: uint32(maxMB) }), nil
}

func RegisterConnectHandlers(mux *http.ServeMux, store Store, cfg Config) {
    // Public DataCollection service
    data := &ConnectServer{store: store}
    path, handler := dcv1connect.NewDataCollectionServiceHandler(data)
    mux.Handle(path, handler)

    // Admin service (protected by Basic Auth)
    adm := &AdminServer{store: store}
    apath, ahandler := dcv1connect.NewAdminServiceHandler(adm)
    mux.Handle(apath, adminAuth(cfg.AdminPassword, ahandler))

    // Upload service
    up := &UploadServer{store: store}
    upath, uhandler := dcv1connect.NewUploadServiceHandler(up)
    mux.Handle(upath, uhandler)

    // Config service
    conf := &ConfigServer{cfg: cfg, store: store}
    cpath, chandler := dcv1connect.NewConfigServiceHandler(conf)
    mux.Handle(cpath, chandler)
}
