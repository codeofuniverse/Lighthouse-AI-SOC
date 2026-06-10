##! Lighthouse Zeek site policy — the rich second sensor.
##!
##! Produces JSON conn.log + http.log with the fields the ML models need that
##! Suricata flow records don't expose. Joined to Suricata events by community-id.

# JSON output so detection/zeek_bridge.py can json.loads each line (same tailing
# path as the Suricata eve.json bridge). The tuning script is the documented,
# load-order-safe way to switch the ASCII writer to JSON (a bare redef can run
# before the logging framework initialises in -i mode).
@load policy/tuning/json-logs.zeek

# Load the core protocol analyzers (conn, http, ssl, dns) — base scripts.
@load base/protocols/conn
@load base/protocols/http
@load base/protocols/ssl
@load base/protocols/dns

# Add the community-id field to conn.log so a Zeek flow can be matched to the
# Suricata event for the same flow (cross-sensor correlation bonus in the
# risk scorer, analogous to Wazuh host+network correlation).
@load policy/protocols/conn/community-id-logging
@load policy/frameworks/notice/community-id

# ── Custom enrichment: TTL + TCP base-seq for the UNSW-28 feature set ─────────
# sttl/dttl/stcpb/dtcpb are not in default conn.log. Capture them from the first
# packets of each connection so unsw28_from_zeek() can read them off the record.
redef record Conn::Info += {
    sttl:  count &log &optional;
    dttl:  count &log &optional;
    stcpb: count &log &optional;
    dtcpb: count &log &optional;
};

event new_packet(c: connection, p: pkt_hdr)
    {
    if ( ! p?$ip )
        return;
    if ( c$id$orig_h == p$ip$src )
        {
        if ( ! c$conn?$sttl )
            c$conn$sttl = p$ip$ttl;
        if ( p?$tcp && ! c$conn?$stcpb )
            c$conn$stcpb = p$tcp$seq;
        }
    else
        {
        if ( ! c$conn?$dttl )
            c$conn$dttl = p$ip$ttl;
        if ( p?$tcp && ! c$conn?$dtcpb )
            c$conn$dtcpb = p$tcp$seq;
        }
    }
