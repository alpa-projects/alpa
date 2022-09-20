RFC: Alpa API Authentication
=================

**Summary:** Provide API authentication in existing Alpa network.

**Status:** WIP | **In-Review** | Approved | Obsolete



# Background:

Issue on [#700 OPT-175B service authentication and new priority queue](https://github.com/alpa-projects/alpa/issues/700)


# Constraints:

1.  No downtime or minimum downtime that’s approved by Hao
    
2.  Protocols and security policies agreed by Qirong and the MBZUAI team
    
3.  Hao wants to have the capability ASAP: “cuz we want to announce a client lib feature and invite more API users”
    

# Proposal:


## Service Architecture

The following proposals are based on the current Alpa/MBZUAI Infra.

In the first 2 options, the auth will be handled by APISIX Gateway, which is deployed with Rampart on Kubernetes.

1.  Intra-cluster: in the existing Alpa nodes, deploy microk8s and Rampart with APISIX. The alpa service can be treated as another upstream service for APISIX Gateway. This would require exposure of APISIX Gateway as external entry point for end users to access Alpa APIs.
    
2.  Inter-cluster: assign additional nodes (3 for high availability) to deploy k8s/Rampart with APISIX. The Alpa service will be treated as an external service for APISIX Gateway. The APISIX Gateway will still be the external entry point for end users to access Alpa APIs.
    
3.  Cloudflare service token: this requires [alpa.ai](http://alpa.ai) domain to be controlled by Cloudflare instead of Azure, which is trivial to do. Rampart is not needed for this option.
    

For option 1 and 2, we can either stay with Azure or switch to Cloudflare to expose the service.


### Comparison Table

|     |     |     |     |
| --- | --- | --- | --- |
| **Option** | **Architecture** | **Pros** | **Cons** |
| Intra-cluster A<br> | <br>Rampart k8s cluster is deployed along side the existing OPT ray cluster on the same DGX nodes |  *   Auth and traffic management options are flexible and powerful   <br>*   Auth components are HA and scalable    <br>*   No extra hardware required    <br>*   No redeployment of OPT ray cluster   <br>*   Minimum downtime | *   Ray cluster itself is not taking advantage of k8s production ready features such as auto restart of failed ray cluster nodes (including head node)<br>    <br>*   Ray cluster itself is not taking advantage of Rampart’s convenience for easy ray cluster upgrade |
| Intra-cluster B (transitioned from A)<br> | <br>OPT ray cluster is deployed as a Rampart component | *   All pros from above except downtime.   <br>*   Ray cluster itself is managed by Rampart for more reliable operation (auto healing/restart of ray nodes) and convenience (upgrade code and models) | *   Some downtime is expected (< 1 hour) to convert the baremetal ray cluster to ray component on rampart (and k8s).   <br>*   Rollback to the previous option might be needed if we run into something unexpected and couldn’t finish the conversion within expected time window, as we might have only tested a smaller model (60B) on a similar (but not the same) k8s DGX cluster. |
| Inter-cluster | Rampart cluster is deployed outside of OPT ray cluster  | *   Pros of the first option except extra hardware requirement. | *   Extra CPU nodes are needed for the rampart cluster to host the APISIX components   <br>*   3+ nodes are needed for HA |
| Cloudflare service token  | Switch Azure to CloudFlare on existing alpa network | *   Pros of the first option except extra flexibility (rich functionality that are only available via the extensive plugins for APISIX gateway)    <br>*   Domain owner has better control over the external endpoint configurations   <br>*   Least effort to implement for short term limited auth (api key/token) needs. | *   A conversion from Azure to Cloudflare is needed.    <br>*   Cons of the first option    <br>*   Some desired features, such as rate throttling based on auth info might not be readily available as previous options |



## API Authentication Methods

We propose 3 options to achieve API authentication for Alpa. The first 2 approaches are based on APISIX Gateway with some configuration, which implies proper deployment of Rampart/APISIX on k8s and proper setup for APISIX.

1.  Key-auth with APISIX
    
2.  Jwt-auth with APISIX
    
3.  Service token with Cloudflare
    

All 3 options don’t require modification to OPT serving code.

### API Key with `key-auth`

In this approach, Admin needs to create a `consumer` (user) in APISIX, giving it a secret key string. The auth is handled on the ingress/route level, as this is an authentication mechanism instead of RBAC/ACL scheme.

The Alpa API consumer/user will then access the API by including a `apikey` header.

```
curl http://10.1.96.188:30800/alpa/ -H 'apikey: user1-secret-apikey' 
```

### API Key with `jwt-auth`

In this approach, Admin needs to create a `consumer` (user) in APISIX with either a HS256-based secret key or RS256-based public/private key pair. The auth is handled at the ingress/route level, as this is an authentication mechanism instead of RBAC/ACL scheme.

The Alpa API consumer/user will then access the API using the generated JWT token.

```
curl http://10.1.96.188:30800/alpa/ -H 'Authorization: <api-token>'
```

### Service Token with Cloudflare

For this option, an admin needs to have access to DNS configuration of [alpa.ai](http://alpa.ai). A (free) cloudflare account is also assumed. Using the Cloudflare Access UI, the admin can create (unlimited?) service tokens for (unlimited) service under the domain. Here is a working example:

```
curl -d '{"prompt": "alpa is cool", "max_tokens": "32", "temperature": "0.4", "top_p": "0.3"}' -H 'Content-type: application/json' \
  -H 'CF-Access-Client-Id: <cf-id>.access' \
  -H 'CF-Access-Client-Secret: <cf-secret>' \
  https://api.alpa.com/alpa-opt-service/completions
{"choices":[{"text":", but i think it's a bit too much of a stretch to say that it's a good game.\nI agree. I think it's a good"}],"created":1663217337,"id":"6f7e5949-64b3-4b02-8de3-9d7baf88e834","object":"text_completion"}
```

# Implementation and Timeline

The general implementation involves either the Rampart approach (option #1 or #2 above) or the CloudFlare approach (#3 above).

## Rampart/Apisix For Alpa

Implementation for this approach requires:

*   allocation of nodes in Alpa network
    
*   deployment of Kubernetes (or microk8s) on these managed nodes 
    
*   deployment of Rampart (with Apisix) in the k8s cluster
    
*   infra level implementation and configuration of Apisix Gateway
    
*   e2e flow testing with external access
    
*   creating consumers/users for targeted users by Admin
    
*   generating proper secret key or tokens and distribute to target users by Admin
 
*   many of the above steps can be done on parallel while Alpa is running as is, the real downtime involves switching Alpa to Rampart infra and necessary testing which can be kept to minimum if planned well
    

## Cloudflare Service Tokens For Alpa

*   Install `cloudflared` on the head node of the ALPA ray cluster
    
*   Configure service token access with Cloudflare SaaS.
    
*   Contribute to ALPA client library for Cloudflare service token integration, which is generally useful.
    
*   Server side time estimate: 1 day (1 hour configuration, remaining for testing and DNS move).
    
*   Client library time estimate: 3 days (1 day coding, 2 day testing and review).
    

Zero down time is doable if the DNS move is planned appropriately.



# Useful Links:

*   Code that Hao suggested to look at:  [GitHub code](https://github.com/alpa-projects/alpa/blob/main/examples/opt_serving/interactive_hosted.py)